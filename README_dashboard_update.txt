Updated ARU dashboard files.

Copy these into ~/aru-dashboard, preserving static/ and templates/. The dashboard expects the acoustic pipeline scripts from aru_pipeline_scripts_with_acquire.zip in ~/aru-dashboard/pipeline unless you edit config.yaml:pipeline.script_dir.

The Calibrate and Localize buttons call acquire.py first. acquire.py reuses existing clips when valid, rotates if the target timestamp is still in an active/growing file, and fetches missing clips.
