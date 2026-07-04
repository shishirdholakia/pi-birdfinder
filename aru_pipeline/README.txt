ARU TXT/H5 pipeline scripts
===========================

Files
-----
rotate_fetch_clips_txt.py    Rotate units, fetch clipped FLAC/tracking files, write TXT manifest/metadata.
aru_calibrate_session.py     Create or update a station-offset calibration TXT from a known calibration clap.
aru_localize_event.py        Localize an unknown event using fetched clips and optional station-offset calibration TXT.
aru_io.py                    Shared event-dir, TXT parsing, locations, audio, simple/Bayesian clock wrapper.
align.py                     Impulse/birdcall filtering, pick, GCC-PHAT alignment.
tdoa.py                      TDOA construction and correction conventions.
calibration.py               Geometry/timing calibration fitting and calibration TXT writer.
localize.py                  2D TDOA localization, MC uncertainty, leave-one-out.
maps.py                      Satellite Folium maps, station-fit and final localization maps.
viz.py                       Waveform and shifted/unshifted spectrogram diagnostics.

Installation assumptions
------------------------
Put these scripts in the same directory as bayesian_clock_map_scipy.py if you want the
Bayesian clock-map wrapper to be used. If that import fails, the scripts fall back to a
simple robust linear tracking map so you can still run field diagnostics.

Python packages:
  numpy scipy soundfile matplotlib folium pyyaml h5py

Typical workflow
----------------
1. Fetch a calibration clap at unit five:
   python rotate_fetch_clips_txt.py "2026-05-18 20:00:30" --clip-half-s 10 --units zero one four five

2. Create/update calibration. Default source is a clap at five:
   python aru_calibrate_session.py /path/to/event_2026-05-18_20-00-30 \
     --calibration-txt /path/to/station_offset_calibration.txt

   If the clap was at another unit:
   python aru_calibrate_session.py /path/to/event_dir --source-unit zero \
     --calibration-txt /path/to/station_offset_calibration.txt

   If the clap was at a coordinate:
   python aru_calibrate_session.py /path/to/event_dir --source-lat 37.0 --source-lon -122.0 \
     --calibration-txt /path/to/station_offset_calibration.txt

3. Fetch an unknown event:
   python rotate_fetch_clips_txt.py "2026-05-18 21:13:42" --clip-half-s 30 --units zero one four five

4. Localize it:
   python aru_localize_event.py /path/to/event_2026-05-18_21-13-42 \
     --calibration-txt /path/to/station_offset_calibration.txt --mode impulse

Outputs
-------
Calibration:
  calibration_output/station_fit_map.html
  calibration_output/diagnostics/waveforms_*.png
  calibration_output/diagnostics/spectrograms_*.png
  data/calibrations/station_offset_calibrations/session_calibration.txt

Localization:
  localization_output/localization_map.html
  localization_output/event_tdoa.csv
  localization_output/localization_summary.txt
  localization_output/diagnostics/waveforms_*.png
  localization_output/diagnostics/spectrograms_*.png

Notes
-----
- Station-offset calibration TXTs are created/updated under data/calibrations/station_offset_calibrations by default.
- The event localizer automatically uses data/calibrations/station_offset_calibrations/joint_station_offsets_manual_override_16-06_16-17_calibration.txt when it exists and no --calibration-txt is supplied. Use --no-calibration for an uncalibrated run.
- Relative calibration paths resolve from the dashboard root, which defaults to the parent directory of aru_pipeline. Set ARU_DASHBOARD_ROOT when launching scripts from another install layout.
- With one calibration clap, station geometry is strongly prior-limited. Multiple claps at known, separated
  source positions are needed to meaningfully refine station coordinates.
- Folium maps use Esri satellite imagery and add contours/samples first, then hyperbolas, then markers/locations.

Acquisition integration update
------------------------------
This bundle now includes acquire.py, which is the idempotent acquisition layer
intended for the dashboard and for timestamp-first calibration/localization
workflows.

Recommended dashboard/backend pattern:

  from acquire import ensure_event_clips

  result = ensure_event_clips(
      "2026-05-06 20:56:00",
      units=["zero", "one", "four", "five"],
      clip_half_s=30.0,
      allow_rotate_if_active=True,
      reuse_existing=True,
  )

  if result.ok:
      event_dir = result.event_dir
      # Then pass event_dir to aru_calibrate_session.py or aru_localize_event.py.

Command-line usage:

  python acquire.py "2026-05-06 20:56:00" --clip-half-s 30 --units zero,one,four,five

Behavior:

  1. Computes the canonical event directory under data/event_clips.
  2. Reuses complete existing clips if they already cover the requested window.
  3. If clips are missing, inspects units over SSH/local transport.
  4. If the target appears to be in a still-active/growing recording, rotates
     all selected units, waits for finalization, and then fetches.
  5. If the target is already in finalized files, skips rotation and fetches
     only missing/invalid unit clips.
  6. Writes acquire_manifest.txt in the event directory.

Suggested timestamp-first shell workflow:

  EVENT_DIR=$(python acquire.py "2026-05-06 20:56:00" --clip-half-s 10 | awk -F' = ' '/^event_dir/{print $2}')
  python aru_calibrate_session.py "$EVENT_DIR" --source-unit five

  EVENT_DIR=$(python acquire.py "2026-05-06 21:12:30" --clip-half-s 30 | awk -F' = ' '/^event_dir/{print $2}')
  python aru_localize_event.py "$EVENT_DIR" --mode impulse

For the FastAPI dashboard, the Calibrate and Localize buttons should call
ensure_event_clips() first, then launch the corresponding processing script as a
background job. Keep a separate "Acquire only" advanced button for debugging.
