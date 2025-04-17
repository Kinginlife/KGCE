from kgce import EnvironmentConfig
from kgce.actions.android_actions import (
    key_press,
    long_tap,
    open_app_drawer,
    screenshot,
    setup,
    swipe,
    tap,
    write_text,
)

ANDROID_ENV = EnvironmentConfig(
    name="android",
    action_space=[tap, key_press, long_tap, write_text, swipe, open_app_drawer],
    observation_space=[screenshot],
    description="""A Google Pixel smartphone runs on the Android operating system. \
The interface displays a current screenshot at each step and primarily \
supports interaction through tapping and typing. This device offers a suite \
of standard applications including Phone, Photos, Camera, Chrome, and \
Calendar, among others. Access the app drawer to view all installed \
applications on the device. The Google account is pre-logged in, synchronized \
with the same account used in the Ubuntu environment.""",
    extra_attributes={"device": None},
    reset=setup,
)
