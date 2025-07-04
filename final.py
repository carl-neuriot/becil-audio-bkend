from scipy.signal import correlate
import time
import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from datetime import timedelta, datetime
import librosa
import librosa.display
from sklearn.preprocessing import normalize
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from models import Ad, Song, Broadcast, AdDetectionResult, ExcelReports
from io import BytesIO
from fastapi.responses import StreamingResponse
import soundfile as sf
from pydub import AudioSegment
from fastapi import HTTPException
import logging
import traceback
import io


def seconds_to_standard_time(seconds):
    return str(timedelta(seconds=seconds)).split('.')[0]


def extract_brand_name(filename):
    if '_' in filename:
        return filename.split('_')[0]
    else:
        return os.path.splitext(filename)[0]


def load_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050, mono=True)
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path} with librosa: {e}")
        try:
            sr, audio = wavfile.read(file_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            if audio.dtype != np.float32 and audio.dtype != np.float64:
                audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
            if sr != 22050:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
                sr = 22050
            return audio, sr
        except Exception as e2:
            print(f"Error loading {file_path} with scipy: {e2}")
            return None, None


def preprocess_audio(audio, sr):
    """Preprocess audio for better matching"""
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    pre_emphasis = 0.97
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    nyquist = sr / 2
    low = 300 / nyquist
    high = 3400 / nyquist

    if low < 1.0 and high < 1.0:
        b, a = signal.butter(4, [low, high], btype='band')
        audio = signal.filtfilt(b, a, audio)

    return audio


def extract_mfcc_features(audio, sr, n_mfcc=13):
    """Extract MFCC features for better audio matching"""
    mfccs = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    features = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
    return features


def compute_feature_correlation(master_features, recording_features):
    """Compute correlation between feature vectors"""
    master_norm = normalize(master_features, axis=0)
    recording_norm = normalize(recording_features, axis=0)

    correlations = []
    for i in range(master_norm.shape[0]):
        corr = signal.correlate(recording_norm[i], master_norm[i], mode='full')
        correlations.append(corr)

    avg_correlation = np.mean(correlations, axis=0)
    return avg_correlation


def normalize_signal(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-10)


def find_matches_improved(master_audio, master_sr, radio_audio, radio_sr, threshold=0.65):
    """
    Improved ad detection using normalized cross-correlation.
    Works with .mp3 or .wav master files.
    """

    # Step 1: Resample ad audio to match radio sampling rate
    if master_sr != radio_sr:
        master_audio = librosa.resample(
            master_audio, orig_sr=master_sr, target_sr=radio_sr)
        master_sr = radio_sr

    # Step 2: Normalize both signals
    master_audio = normalize_signal(master_audio)
    radio_audio = normalize_signal(radio_audio)

    # Step 3: Cross-correlation
    correlation = correlate(radio_audio, master_audio, mode='valid')
    correlation /= len(master_audio)

    matches = []
    ad_duration = len(master_audio) / radio_sr

    i = 0
    while i < len(correlation):
        if correlation[i] >= threshold:
            start_time = i / radio_sr
            end_time = start_time + ad_duration
            matches.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': ad_duration,
                'correlation': float(round(correlation[i], 4))
            })
            # Skip forward by ad duration to avoid overlap
            i += int(ad_duration * radio_sr)
        else:
            i += 1

    return matches


Base = declarative_base()


class EnhancedRadioRecordingManager:
    def __init__(self, db_path="radio_ad_detection.db"):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.Session = Session

    def get_ad_id_by_filename(self, filename):
        """Get ad ID from the ads table by filename"""
        session = self.Session()
        try:
            ad = session.query(Ad).filter(
                Ad.advertisement == filename).first()
            return ad.id if ad else -1
        except Exception as e:
            print(f"Error getting ad ID for {filename}: {e}")
            return -1
        finally:
            session.close()

    def get_broadcast_by_id(self, broadcast_id):
        session = self.Session()
        try:
            broadcast = session.query(Broadcast).filter(
                Broadcast.id == broadcast_id).first()
            return broadcast
        except Exception as e:
            print(f"Error getting broadcast for id {broadcast_id}: {e}")
            return -1
        finally:
            session.close()

    def get_broadcast_id_by_filename(self, filename):
        """Get broadcast ID from the broadcasts table by filename"""
        session = self.Session()
        try:
            broadcast = session.query(Broadcast).filter(
                Broadcast.broadcast_recording == filename).first()
            return broadcast.id if broadcast else -1
        except Exception as e:
            print(f"Error getting broadcast ID for {filename}: {e}")
            return -1
        finally:
            session.close()

    def populate_ads_from_folder(self, ad_masters_folder="ad_masters"):
        """Populate ads table from ad_masters folder"""
        session = self.Session()
        try:
            print(f" Scanning {ad_masters_folder} for advertisement files...")

            if not os.path.exists(ad_masters_folder):
                print(f"✗ Folder {ad_masters_folder} does not exist")
                return 0

            added_count = 0
            updated_count = 0

            for filename in os.listdir(ad_masters_folder):
                if filename.endswith(('.wav', '.mp3')):
                    filepath = os.path.join(ad_masters_folder, filename)

                    # Check if ad already exists
                    existing_ad = session.query(Ad).filter(
                        Ad.advertisement == filename
                    ).first()

                    # Get duration
                    duration_seconds = None
                    audio, sr = load_audio(filepath)
                    if audio is not None:
                        duration_seconds = int(len(audio) / sr)

                    brand_name = extract_brand_name(filename)

                    if existing_ad:
                        # Update existing record
                        existing_ad.brand = brand_name
                        existing_ad.duration = duration_seconds
                        existing_ad.status = 'Active'
                        updated_count += 1
                    else:
                        # Add new record
                        new_ad = Ad(
                            brand=brand_name,
                            advertisement=filename,
                            duration=duration_seconds,
                            status='Active'
                        )
                        session.add(new_ad)
                        added_count += 1

            session.commit()
            print(" Ad table updated:")
            print(f"  - New ads added: {added_count}")
            print(f"  - Existing ads updated: {updated_count}")

            return added_count + updated_count

        except Exception as e:
            session.rollback()
            print(f"✗ Error populating ads table: {e}")
            return 0
        finally:
            session.close()

    def get_all_ads(self):
        """Get all ads from the database"""
        session = self.Session()
        try:
            ads = session.query(Ad).order_by(
                Ad.brand, Ad.advertisement).all()

            if not ads:
                print("No ads found in database.")
                return []

            print(f"\n Advertisement Masters ({len(ads)}):")
            print("=" * 80)

            ads_info = []
            for i, ad in enumerate(ads, 1):
                duration_str = seconds_to_standard_time(
                    ad.duration) if ad.duration else "Unknown"
                info = {
                    'id': ad.id,
                    'brand': ad.brand,
                    'advertisement': ad.advertisement,
                    'duration': ad.duration,
                    'duration_str': duration_str,
                    'upload_date': ad.upload_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'status': ad.status
                }
                ads_info.append(info)

                print(f"{i:2d}. {ad.brand} - {ad.advertisement} (ID: {ad.id})")
                print(
                    f"     Duration: {duration_str} | Status: {ad.status} | Uploaded: {info['upload_date']}")
                print()

            return ads_info

        except Exception as e:
            print(f"Error getting ads: {e}")
            return []
        finally:
            session.close()

    def add_broadcast(self, broadcast_recording, radio_station=None, duration=None, broadcast_date=None, status='Pending'):
        """Add a new broadcast to the database"""
        session = self.Session()
        try:
            # Check if broadcast already exists
            existing = session.query(Broadcast).filter(
                Broadcast.broadcast_recording == broadcast_recording
            ).first()

            if existing:
                print(
                    f" Broadcast {broadcast_recording} already exists in database (ID: {existing.id})")
                return existing.id

            if broadcast_date is None:
                broadcast_date = datetime.now()

            new_broadcast = Broadcast(
                radio_station=radio_station,
                broadcast_recording=broadcast_recording,
                duration=duration,
                broadcast_date=broadcast_date,
                status=status
            )

            session.add(new_broadcast)
            session.commit()

            print(
                f" Added broadcast: {broadcast_recording} (ID: {new_broadcast.id})")
            return new_broadcast.id

        except Exception as e:
            session.rollback()
            print(f" Error adding broadcast: {e}")
            return None
        finally:
            session.close()

    def update_broadcast_status(self, broadcast_id, status):
        """Update broadcast processing status"""
        session = self.Session()
        try:
            broadcast = session.query(Broadcast).filter(
                Broadcast.id == broadcast_id
            ).first()

            if broadcast:
                broadcast.status = status
                session.commit()
                print(f" Updated {broadcast_id} status to: {status}")
                return True
            else:
                print(f" Broadcast {broadcast_id} not found")
                return False

        except Exception as e:
            session.rollback()
            print(f" Error updating broadcast status: {e}")
            return False
        finally:
            session.close()

    def get_all_broadcasts(self):
        """Get all broadcasts from the database"""
        session = self.Session()
        try:
            broadcasts = session.query(Broadcast).order_by(
                Broadcast.broadcast_date.desc()).all()

            if not broadcasts:
                print("No broadcasts found in database.")
                return []

            print(f"\n Broadcast Recordings ({len(broadcasts)}):")
            print("=" * 80)

            broadcasts_info = []
            for i, broadcast in enumerate(broadcasts, 1):
                duration_str = seconds_to_standard_time(
                    broadcast.duration) if broadcast.duration else "Unknown"
                info = {
                    'id': broadcast.id,
                    'radio_station': broadcast.radio_station or "Unknown",
                    'broadcast_recording': broadcast.broadcast_recording,
                    'duration': broadcast.duration,
                    'duration_str': duration_str,
                    'broadcast_date': broadcast.broadcast_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'status': broadcast.status
                }
                broadcasts_info.append(info)

                print(
                    f"{i:2d}. {broadcast.broadcast_recording} (ID: {broadcast.id})")
                print(
                    f"     Station: {info['radio_station']} | Duration: {duration_str}")
                print(
                    f"     Date: {info['broadcast_date']} | Status: {broadcast.status}")
                print()

            return broadcasts_info

        except Exception as e:
            print(f"Error getting broadcasts: {e}")
            return []
        finally:
            session.close()

    def save_detection_results(self, broadcast_id, matches_dict, radio_filename):
        """Save detection results to database for a specific radio recording"""
        session = self.Session()

        try:
            # Get broadcast ID for this radio file
            # broadcast_id = self.get_broadcast_id_by_filename(radio_filename)

            if broadcast_id == -1:
                print(
                    f"⚠ Warning: Broadcast {radio_filename} not found in database")
                return 0

            # Clear existing results for this broadcast (for reprocessing)
            session.query(AdDetectionResult).filter(
                AdDetectionResult.broadcast_id == broadcast_id
            ).delete()

            # Process and filter matches
            all_matches = []
            for master_name, matches in matches_dict.items():
                # Get ad ID for this master file
                ad_id = self.get_ad_id_by_filename(master_name)

                for match in matches:
                    all_matches.append({
                        'master_name': master_name,
                        'ad_id': ad_id,
                        'start_time': match['start_time'],
                        'end_time': match['end_time'],
                        'duration': match['duration'],
                        'correlation': match['correlation'],
                        'overlap_duration': match.get('overlap_duration', match['duration']),
                        'raw_correlation': match.get('raw_correlation', match['correlation']),
                        'mfcc_correlation': match.get('mfcc_correlation', match['correlation']),
                        'clip_type': 'ad'
                    })

            # Sort and filter overlapping matches
            all_matches.sort(key=lambda x: x['start_time'])
            final_matches = self._filter_overlapping_matches(all_matches)

            # Save to database
            for match in final_matches:
                brand_name = extract_brand_name(match['master_name'])
                description = os.path.splitext(match['master_name'])[0]

                db_record = AdDetectionResult(
                    brand=brand_name,
                    description=description,
                    start_time_seconds=match['start_time'],
                    end_time_seconds=match['end_time'],
                    duration_seconds=match['duration'],
                    correlation_score=match['correlation'],
                    raw_correlation=match['raw_correlation'],
                    mfcc_correlation=match['mfcc_correlation'],
                    overlap_duration=match['overlap_duration'],
                    total_matches_found=len(final_matches),
                    ad_id=match['ad_id'],
                    broadcast_id=broadcast_id,
                    clip_type=match['clip_type']
                )
                session.add(db_record)

            session.commit()
            print(
                f" Saved {len(final_matches)} detection results for {radio_filename}")
            print(f"   - Broadcast ID: {broadcast_id}")
            print(f"   - Ad IDs linked successfully")

            # Update broadcast status to "Completed" if it exists
            self.update_broadcast_status(broadcast_id, "Completed")

            # Generate and store Excel report
            self._generate_and_store_excel(broadcast_id, final_matches)

            return len(final_matches)

        except Exception as e:
            session.rollback()
            print(f" Error saving results: {e}")
            return 0
        finally:
            session.close()

    def _filter_overlapping_matches(self, all_matches):
        """Apply filtering logic for overlapping matches"""
        final_matches = []
        for match in all_matches:
            should_keep = True

            for i, existing_match in enumerate(final_matches):
                overlap_start = max(
                    match['start_time'], existing_match['start_time'])
                overlap_end = min(match['end_time'],
                                  existing_match['end_time'])
                overlap_duration = max(0, overlap_end - overlap_start)

                min_duration = min(match['duration'],
                                   existing_match['duration'])

                if overlap_duration > (0.4 * min_duration):
                    current_score = match['correlation']
                    existing_score = existing_match['correlation']

                    if current_score > existing_score:
                        final_matches[i] = match
                        should_keep = False
                        break
                    else:
                        should_keep = False
                        break

            if should_keep:
                final_matches.append(match)

        return final_matches

    def _generate_and_store_excel(self, broadcast_id, final_matches):
        """Generate Excel and store in database - WITHOUT correlation score"""
        session = self.Session()
        try:
            # Get broadcast info
            broadcast = session.query(Broadcast).filter(
                Broadcast.id == broadcast_id).first()
            if not broadcast:
                print(f" Broadcast with ID {broadcast_id} not found")
                return

            # Create Excel in memory
            output = BytesIO()

            # Prepare data - REMOVED correlation score column
            data = []
            header_data = {
                'Brand': 'Brand',
                'Description': 'Description',
                'Start Time (HH:MM:SS)': 'Start Time (HH:MM:SS)',
                'End Time (HH:MM:SS)': 'End Time (HH:MM:SS)',
                'Ad Duration (HH:MM:SS)': 'Ad Duration (HH:MM:SS)'
            }
            data.append(header_data)

            for match in final_matches:
                start_rounded = max(0, round(match['start_time']))
                end_rounded = round(match['end_time'])
                duration_rounded = end_rounded - start_rounded

                data.append({
                    'Brand': extract_brand_name(match['master_name']),
                    'Description': os.path.splitext(match['master_name'])[0],
                    'Start Time (HH:MM:SS)': seconds_to_standard_time(start_rounded),
                    'End Time (HH:MM:SS)': seconds_to_standard_time(end_rounded),
                    'Ad Duration (HH:MM:SS)': seconds_to_standard_time(abs(duration_rounded))
                })

            df = pd.DataFrame(data)

            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Ad Detection Results',
                            index=False, header=False)

                worksheet = writer.sheets['Ad Detection Results']

                from openpyxl.styles import PatternFill, Font, Alignment

                yellow_fill = PatternFill(
                    start_color="FFFF00", end_color="FFFF00", fill_type="solid")
                bold_font = Font(bold=True)
                center_alignment = Alignment(horizontal="center")

                # Format header row
                for col in range(1, len(df.columns) + 1):
                    cell = worksheet.cell(row=1, column=col)
                    cell.fill = yellow_fill
                    cell.font = bold_font
                    cell.alignment = center_alignment

                # Adjust column widths - REMOVED correlation score column width
                worksheet.column_dimensions['A'].width = 20
                worksheet.column_dimensions['B'].width = 60
                worksheet.column_dimensions['C'].width = 18
                worksheet.column_dimensions['D'].width = 18
                worksheet.column_dimensions['E'].width = 20

            excel_data = output.getvalue()

            # Store Excel in database
            excel_filename = f"detection_results_{broadcast.broadcast_recording.replace('.mp3', '').replace('.wav', '')}.xlsx"

            # Remove existing Excel report for this broadcast
            session.query(ExcelReports).filter(
                ExcelReports.broadcast_id == broadcast_id
            ).delete()

            excel_record = ExcelReports(
                broadcast_id=broadcast_id,
                excel_data=excel_data,
                excel_filename=excel_filename,
                total_ads_detected=len(final_matches),
                file_size_bytes=len(excel_data)
            )
            session.add(excel_record)
            session.commit()

            print(
                f" Excel report stored in database for broadcast ID {broadcast_id}")

        except Exception as e:
            session.rollback()
            print(f" Error storing Excel: {e}")
        finally:
            session.close()

    def get_radio_recordings_list(self):
        """Get list of all processed radio recordings"""
        session = self.Session()
        try:
            broadcasts = session.query(Broadcast.broadcast_recording).all()
            return [b[0] for b in broadcasts]
        except Exception as e:
            print(f"Error getting recordings list: {e}")
            return []
        finally:
            session.close()

    def download_excel_by_id(self, broadcast_id, save_path=None):
        """Download Excel file from database by radio filename"""
        session = self.Session()
        try:
            # Get broadcast ID first
            broadcast = session.query(Broadcast).filter(
                Broadcast.id == broadcast_id
            ).first()

            if not broadcast:
                print(f"No broadcast found for: {broadcast_id}")
                return None

            excel_record = session.query(ExcelReports).filter(
                ExcelReports.broadcast_id == broadcast_id
            ).first()

            if not excel_record:
                print(f"No Excel report found for: {broadcast_id}")
                return None

            if save_path is None:
                save_path = excel_record.excel_filename

            with open(save_path, 'wb') as f:
                f.write(excel_record.excel_data)

            print(f" Excel downloaded: {save_path}")
            print(f"  - Total ads detected: {excel_record.total_ads_detected}")
            print(f"  - File size: {excel_record.file_size_bytes} bytes")
            print(f"  - Created: {excel_record.created_timestamp}")

            return save_path

        except Exception as e:
            print(f" Error downloading Excel: {e}")
            return None
        finally:
            session.close()

    def stream_excel_report(self, broadcast_id):
        """Return Excel file as a FastAPI StreamingResponse"""
        session = self.Session()
        try:
            broadcast = session.query(Broadcast).filter(
                Broadcast.id == broadcast_id
            ).first()

            if not broadcast:
                print(f"No broadcast found for: {broadcast_id}")
                return None

            excel_record = session.query(ExcelReports).filter(
                ExcelReports.broadcast_id == broadcast_id
            ).first()

            if not excel_record:
                print(f"No Excel report found for: {broadcast_id}")
                return None

            file_stream = BytesIO(excel_record.excel_data)
            filename = excel_record.excel_filename or f"Report_{broadcast.broadcast_recording}.xlsx"

            return StreamingResponse(
                file_stream,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )

        except Exception as e:
            print(f" Error streaming Excel: {e}")
            return None
        finally:
            session.close()

    def get_all_available_reports(self):
        """Get information about all available Excel reports"""
        session = self.Session()
        try:
            reports = session.query(ExcelReports, Broadcast).join(
                Broadcast, ExcelReports.broadcast_id == Broadcast.id
            ).order_by(ExcelReports.created_timestamp.desc()).all()

            if not reports:
                print("No reports available in database.")
                return []

            print(f"\n Available Reports ({len(reports)}):")
            print("=" * 80)

            report_info = []
            for i, (report, broadcast) in enumerate(reports, 1):
                info = {
                    'id': report.id,
                    'radio_file': broadcast.broadcast_recording,
                    'excel_filename': report.excel_filename,
                    'ads_detected': report.total_ads_detected,
                    'created': report.created_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'size_kb': round(report.file_size_bytes / 1024, 2),
                    'broadcast_id': broadcast.id
                }
                report_info.append(info)

                print(f"{i:2d}. {broadcast.broadcast_recording}")
                print(f"     Excel: {report.excel_filename}")
                print(
                    f"     Ad: {report.total_ads_detected} | Created: {info['created']} | Size: {info['size_kb']} KB")
                print()

            return report_info

        except Exception as e:
            print(f"Error getting reports: {e}")
            return []
        finally:
            session.close()

    def get_detection_summary(self, radio_filename):
        """Get detailed summary of detection results"""
        session = self.Session()
        try:
            # Get broadcast first
            broadcast = session.query(Broadcast).filter(
                Broadcast.broadcast_recording == radio_filename
            ).first()

            if not broadcast:
                print(f" No broadcast found for: {radio_filename}")
                return None

            results = session.query(AdDetectionResult).filter(
                AdDetectionResult.broadcast_id == broadcast.id
            ).order_by(AdDetectionResult.start_time_seconds.asc()).all()

            if not results:
                print(f" No results found for: {radio_filename}")
                return None

            print(f"\n Detection Summary for: {radio_filename}")
            print("=" * 60)
            print(f"Total Ad Detected: {len(results)}")
            print(
                f"Processing Date: {results[0].detection_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Broadcast ID: {results[0].broadcast_id}")
            print()

            # Brand summary
            brand_counts = {}
            total_duration = 0

            for result in results:
                brand_counts[result.brand] = brand_counts.get(
                    result.brand, 0) + 1
                total_duration += result.duration_seconds

            print(" Brand Breakdown:")
            for brand, count in sorted(brand_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {brand}: {count} ads")

            print(
                f"\n Total Ad Duration: {seconds_to_standard_time(total_duration)}")
            print(
                f" Average Correlation Score: {np.mean([r.correlation_score for r in results]):.4f}")

            # Show first few results with IDs
            print(f"\n First 5 Detections (with IDs):")
            for i, result in enumerate(results[:5], 1):
                print(
                    f"  {i}. {result.brand} | Ad ID: {result.ad_id} | {seconds_to_standard_time(result.start_time_seconds)}")

            return {
                'total_ads': len(results),
                'brands': brand_counts,
                'total_duration': total_duration,
                'avg_correlation': np.mean([r.correlation_score for r in results]),
                'broadcast_id': results[0].broadcast_id
            }

        except Exception as e:
            print(f"Error getting summary: {e}")
            return None
        finally:
            session.close()

    def extract_clip_from_broadcast(
            self,
            broadcast_id,
            brand_or_artist,
            advertisement_or_name,
            clip_type,
            start_time,
            end_time,
            ad_masters_folder="ad_masters",
            songs_folder="songs"
    ):
        session = self.Session()
        start_time = time.time()
        try:
            # Get broadcast information
            broadcast = session.query(Broadcast).filter(
                Broadcast.id == broadcast_id
            ).first()

            if not broadcast:
                print(f" Broadcast with ID {broadcast_id} not found")
                return False

            broadcast_file = broadcast.broadcast_recording
            print(f" Processing broadcast: {broadcast_file}")

            if clip_type.lower() == 'speech':
                print(" Speech type detected - inserting directly to database")

                speech_result = AdDetectionResult(
                    brand="Speech",
                    description="",
                    start_time_seconds=float(start_time),
                    end_time_seconds=float(end_time),
                    duration_seconds=float(end_time - start_time),
                    correlation_score=1.0,  # Default score for manually added speech
                    raw_correlation=1.0,
                    mfcc_correlation=1.0,
                    overlap_duration=float(end_time - start_time),
                    total_matches_found=1,
                    ad_id=-1,  # No corresponding ad file
                    broadcast_id=broadcast_id,
                    processing_status='manual_entry',
                    clip_type='speech'

                )

                session.add(speech_result)
                session.commit()
                print(f" Speech entry added to detection results")
                return True

        # Load the broadcast audio
            radio_file_path = get_first_file_path("radio_recording")
            broadcast_audio, broadcast_sr = load_audio(radio_file_path)
            if broadcast_audio is None:
                print(f" Failed to load broadcast audio")
                return False
            else:
                print("Loaded audio file")

        # Calculate sample indices for extraction
            start_sample = int(start_time * broadcast_sr)
            end_sample = int(end_time * broadcast_sr)

        # Validate indices
            if start_sample < 0 or end_sample > len(broadcast_audio) or start_sample >= end_sample:
                print(f" Invalid time range: {start_time}s to {end_time}s")
                print(
                    f"   Broadcast duration: {len(broadcast_audio)/broadcast_sr:.2f}s")
                return False

        # Extract the audio clip
            extracted_clip = broadcast_audio[start_sample:end_sample]
            duration = len(extracted_clip) / broadcast_sr

            print(
                f" Extracted clip: {duration:.2f}s ({start_time}s - {end_time}s)")

        # Generate filename based on type
            if clip_type.lower() == 'ad':
                # Create ad filename: Brand_Description_duration.wav
                safe_brand = "".join(
                    c for c in brand_or_artist if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_desc = "".join(c for c in advertisement_or_name if c.isalnum() or c in (
                    ' ', '-', '_')).rstrip()
                filename = f"{safe_brand}_{safe_desc}.mp3"
                output_folder = ad_masters_folder

            elif clip_type.lower() == 'song':
                # Create song filename: Artist_SongName_duration.wav
                safe_artist = "".join(
                    c for c in brand_or_artist if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_song = "".join(c for c in advertisement_or_name if c.isalnum() or c in (
                    ' ', '-', '_')).rstrip()
                filename = f"{safe_artist}_{safe_song}.mp3"
                output_folder = songs_folder

            else:
                print(
                    f" Invalid clip type: {clip_type}. Use 'ad', 'song', or 'speech'")
                return False

        # Create output directory if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, filename)

        # Save the extracted clip
            try:
                # Convert numpy array to AudioSegment
                # First save as temporary WAV in memory
                temp_wav = io.BytesIO()
                sf.write(temp_wav, extracted_clip, broadcast_sr, format='WAV')
                temp_wav.seek(0)

    # Load into pydub and export as MP3
                audio_segment = AudioSegment.from_wav(temp_wav)
                audio_segment.export(output_path, format="mp3", bitrate="192k")
                print(f" Saved {clip_type} clip: {output_path}")
            except Exception as e:
                print(f" Error saving audio file: {e}")
                return False

        # Add to appropriate database table
            if clip_type.lower() == 'ad':
                # Add to ads table
                new_ad = Ad(
                    brand=brand_or_artist,
                    advertisement=filename,
                    duration=int(duration),
                    status='Active'
                )
                session.add(new_ad)
                session.commit()

            # Get the new ad ID
                ad_id = new_ad.id
                print(f" Added to ads table (ID: {ad_id})")

            # Also add to detection results to show it was found in this broadcast
                detection_result = AdDetectionResult(
                    brand=brand_or_artist,
                    description=advertisement_or_name,
                    start_time_seconds=float(start_time),
                    end_time_seconds=float(end_time),
                    duration_seconds=duration,
                    correlation_score=1.0,  # Perfect score for manually extracted
                    raw_correlation=1.0,
                    mfcc_correlation=1.0,
                    overlap_duration=duration,
                    total_matches_found=1,
                    ad_id=ad_id,
                    broadcast_id=broadcast_id,
                    processing_status='manual_extraction',
                    clip_type='ad'

                )
                session.add(detection_result)

            elif clip_type.lower() == 'song':
                # Add to songs table (assuming you want to track songs similarly)
                new_song = Song(
                    artist=brand_or_artist,
                    name=advertisement_or_name,
                    filename=filename,
                    duration=int(duration),
                    upload_date=datetime.now(),
                    status='Active'
                )
                session.add(new_song)
                session.commit()
                song_id = new_song.id
                print(f" Added to songs table: {filename}")
                song_detection_result = AdDetectionResult(
                    brand=brand_or_artist,
                    description=advertisement_or_name,
                    start_time_seconds=float(start_time),
                    end_time_seconds=float(end_time),
                    duration_seconds=duration,
                    correlation_score=1.0,
                    raw_correlation=1.0,
                    mfcc_correlation=1.0,
                    overlap_duration=duration,
                    total_matches_found=1,
                    ad_id=song_id,
                    broadcast_id=broadcast_id,
                    processing_status='manual_extraction',
                    clip_type='song'
                )
                session.add(song_detection_result)
            session.commit()

            elapsed = time.time() - start_time
            logging.info(f"extract_clip_from_broadcast completed in {elapsed:.2f} seconds")

            print(f" Successfully processed {clip_type}: {filename}")
            print(f"    Saved to: {output_folder}")
            print(f"    Duration: {duration:.2f}s")
            print(f"    Linked to broadcast ID: {broadcast_id}")

            return True

        except Exception as e:
            session.rollback()
            print(f" Error extracting clip: {e}")
            error_msg = f"extract_clip_from_broadcast failed after {time.time() - start_time:.2f} seconds: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=505, detail=str(e))


            return False
        finally:
            session.close()

# In[27]:


def get_first_file_path(folder_path: str) -> str | None:
    try:
        files = os.listdir(folder_path)
        files = [f for f in files if os.path.isfile(
            os.path.join(folder_path, f))]
        if not files:
            return None
        first_file = files[0]
        return os.path.abspath(os.path.join(folder_path, first_file))
    except FileNotFoundError:
        return None

# Main processing function for single audio clip


def process_single_radio_clip(broadcast_id, ad_masters_folder="ad_masters", radio_recording_directory="radio_recording", correlation_threshold=0.65, db_manager=None):
    """Process a single radio recording clip and save everything to database"""

    start_time = time.time()
    try:
        if db_manager is None:
            db_manager = EnhancedRadioRecordingManager()

        radio_file_path = get_first_file_path(radio_recording_directory)

        print(f"\n Processing: {radio_file_path}")
        print("=" * 50)

        print(" Loading advertisement masters...")
        masters = {}
        for filename in os.listdir(ad_masters_folder):
            if filename.endswith(('.wav', '.mp3')):
                filepath = os.path.join(ad_masters_folder, filename)
                audio, sr = load_audio(filepath)
                if audio is not None:
                    masters[filename] = {
                        'audio': audio,
                        'sr': sr,
                        'duration': len(audio) / sr
                    }

        print(f" Loaded {len(masters)} advertisement masters")

        # Load radio recording
        print(" Loading radio recording...")
        radio_recording, radio_sr = load_audio(radio_file_path)
        if radio_recording is None:
            print(f" Error: Could not load radio recording")
            db_manager.update_broadcast_status(broadcast_id, "Failed")
            return False

        radio_duration = len(radio_recording) / radio_sr
        print(
            f" Loaded radio recording (Duration: {seconds_to_standard_time(radio_duration)})")

        # Find matches
        print(" Finding advertisement matches...")
        all_matches = {}
        total_matches = 0

        for master_name, master_data in masters.items():
            matches = find_matches_improved(
                master_data['audio'],
                master_data['sr'],
                radio_recording,
                radio_sr,
                threshold=correlation_threshold
            )
            all_matches[master_name] = matches
            total_matches += len(matches)
            if len(matches) > 0:
                print(f"{master_name}: {len(matches)} matches")

        print(f"Total raw matches found: {total_matches}")

        # Save to database (this also generates and stores Excel)
        final_matches = db_manager.save_detection_results(
            broadcast_id, all_matches, radio_file_path)
        db_manager.update_broadcast_status(broadcast_id, 'Processed')

        # Logging
        elapsed = time.time() - start_time
        logging.info(f"process_single_radio_clip completed in {elapsed:.2f} seconds")

        if final_matches > 0:
            print(f" Processing completed successfully!")
            print(f"    Final matches: {final_matches}")
            print(f"    Results saved to database")
            print(f"    Excel report generated and stored")

            # Show summary
            db_manager.get_detection_summary(radio_file_path)

            return True
        else:
            print(f"  No matches found above threshold")
            db_manager.update_broadcast_status(broadcast_id, "No Matches")
            return False

    except Exception as e:
        error_msg = f"process_single_radio_clip failed after {time.time() - start_time:.2f} seconds: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=505, detail=str(e))


def setup_database_tables(ad_masters_folder="ad_masters"):
    """Initialize and populate ads table from ad_masters folder"""
    db_manager = EnhancedRadioRecordingManager()

    print(" Setting up database tables...")

    # Populate ads table
    ads_count = db_manager.populate_ads_from_folder(ad_masters_folder)

    print(f" Database setup complete!")
    print(f"  - Advertisement masters: {ads_count}")

    return db_manager


def view_all_ads():
    """View all advertisement masters in database"""
    db_manager = EnhancedRadioRecordingManager()
    return db_manager.get_all_ads()


def view_all_broadcasts():
    """View all broadcast recordings in database"""
    db_manager = EnhancedRadioRecordingManager()
    return db_manager.get_all_broadcasts()


def add_new_broadcast(broadcast_file, radio_station=None, duration=None):
    """Add a new broadcast to the database"""
    db_manager = EnhancedRadioRecordingManager()
    return db_manager.add_broadcast(broadcast_file, radio_station, duration)


def update_ads_database(ad_masters_folder="ad_masters"):
    """Update ads table with any new advertisement files"""
    db_manager = EnhancedRadioRecordingManager()
    return db_manager.populate_ads_from_folder(ad_masters_folder)

# Convenience functions for database access


def fetch_excel_report(broadcast_id, download_path=None):
    """Fetch Excel report from database by radio filename"""
    db_manager = EnhancedRadioRecordingManager()
    return db_manager.stream_excel_report(broadcast_id)


def list_all_reports():
    """List all available reports in database"""
    db_manager = EnhancedRadioRecordingManager()
    return db_manager.get_all_available_reports()


def get_report_summary(radio_filename):
    """Get detailed summary of a specific report"""
    db_manager = EnhancedRadioRecordingManager()
    return db_manager.get_detection_summary(radio_filename)

# Main execution functions


def process_current_audio_clip(ad_masters_folder="ad_masters", radio_recording_folder="radio_recording",
                               correlation_threshold=0.65):
    """Process the single audio clip currently in radio_recording folder"""

    # Get the single audio file in the folder
    audio_files = []
    for filename in os.listdir(radio_recording_folder):
        if filename.endswith(('.wav', '.mp3')):
            audio_files.append(filename)

    if len(audio_files) == 0:
        print(" No audio files found in radio_recording folder")
        return False
    elif len(audio_files) > 1:
        print(
            f"  Multiple audio files found. Processing the first one: {audio_files[0]}")

    audio_file = audio_files[0]
    audio_path = os.path.join(radio_recording_folder, audio_file)

    # Process the single clip
    success = process_single_radio_clip(
        ad_masters_folder, audio_path, correlation_threshold)

    if success:
        print(f"\n Successfully processed: {audio_file}")
        print(" You can now:")
        print(f"   1. View reports: list_all_reports()")
        print(f"   2. Download Excel: fetch_excel_report('{audio_file}')")
        print(f"   3. Get summary: get_report_summary('{audio_file}')")
        return audio_file
    else:
        print(f"\n Failed to process: {audio_file}")
        return None

# Quick access functions (what others will use)


def download_latest_report(save_folder="downloads"):
    """Download the most recent Excel report"""
    os.makedirs(save_folder, exist_ok=True)
    reports = list_all_reports()
    if reports:
        latest = reports[0]  # Reports are sorted by creation time desc
        save_path = os.path.join(save_folder, latest['excel_filename'])
        return fetch_excel_report(latest['radio_file'], save_path)
    else:
        print("No reports available")
        return None


def download_report_by_radio_name(radio_filename, save_folder="downloads"):
    """Download Excel report by radio filename"""
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(
        save_folder, f"report_{radio_filename.replace('.mp3', '').replace('.wav', '')}.xlsx")
    return fetch_excel_report(radio_filename, save_path)


def extract_clip(broadcast_id, brand_or_artist, advertisement_or_name,
                 clip_type, start_time, end_time):
    """
    Simple wrapper function to extract clips

    Example usage:
    extract_clip(
        broadcast_id=1,
        brand_or_artist="CocaCola", 
        advertisement_or_name="Summer Campaign 2024",
        clip_type="ad",
        start_time=120.5,
        end_time=150.8
    )
    """
    db_manager = EnhancedRadioRecordingManager()
    return db_manager.extract_clip_from_broadcast(
        broadcast_id, brand_or_artist, advertisement_or_name,
        clip_type, start_time, end_time
    )
# COMPREHENSIVE DATABASE OVERVIEW FUNCTION


def show_broadcasts_for_extraction():
    """Show all broadcasts available for clip extraction"""
    db_manager = EnhancedRadioRecordingManager()
    broadcasts = db_manager.get_all_broadcasts()

    if not broadcasts:
        print("No broadcasts available for extraction")
        return

    print("\n🎵 Available Broadcast for Clip Extraction:")
    print("=" * 60)
    for broadcast in broadcasts:
        duration_str = broadcast['duration_str'] if broadcast['duration'] else "Unknown"
        print(f"ID: {broadcast['id']} | {broadcast['broadcast_recording']}")
        print(f"    Duration: {duration_str} | Status: {broadcast['status']}")
        print(f"    Date: {broadcast['broadcast_date']}")
        print()

    return broadcasts


def show_database_overview():
    """Show complete overview of database contents"""
    db_manager = EnhancedRadioRecordingManager()

    print("\n" + "="*80)
    print("  COMPLETE DATABASE OVERVIEW")
    print("="*80)

    # Get counts
    session = db_manager.Session()
    try:
        ads_count = session.query(Ad).count()
        broadcasts_count = session.query(Broadcast).count()
        results_count = session.query(AdDetectionResult).count()
        reports_count = session.query(ExcelReports).count()

        print(f" Advertisement Masters: {ads_count}")
        print(f" Broadcast Recordings: {broadcasts_count}")
        print(f" Detection Results: {results_count}")
        print(f" Excel Reports: {reports_count}")
        print()

        # Show recent activity
        recent_broadcasts = session.query(Broadcast).order_by(
            Broadcast.broadcast_date.desc()).limit(3).all()
        if recent_broadcasts:
            print(" Recent Broadcast:")
            for b in recent_broadcasts:
                print(
                    f"   • {b.broadcast_recording} ({b.status}) - {b.broadcast_date.strftime('%Y-%m-%d')}")
            print()

    except Exception as e:
        print(f"Error getting overview: {e}")
    finally:
        session.close()
