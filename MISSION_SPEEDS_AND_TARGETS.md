# Mission Speeds and Target Handling

Flight and navigation speeds are configured in `config.py`. Values suitable for one aircraft may be unsafe for another.

- `LAP_SPEED_MPS`: speed along the lap route.
- `MAPPING_SPEED_MPS`: speed along the mapping grid.
- `GUIDED_SPEED_MPS`: speed used for GUIDED navigation.
- `TARGET_CENTERING_MAX_SPEED_MPS`: maximum horizontal speed during visual centering.
- `TARGET_CLASS_NAMES` and `PAYLOAD_TARGET_CHANNELS`: model class labels and their payload-servo mapping.

The mission records target observations during mapping, revisits selected target locations, performs visual centering, and actuates the configured payload servos. Confirm target selection and fallback behaviour in SITL with representative video before flight.

**Safety:** keep `AUTO_START_MISSION = False` until all routes, model paths, camera settings, payload channels, failsafes, and altitude limits have been configured and tested.
