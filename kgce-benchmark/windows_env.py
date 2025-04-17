from kgce.actions.desktop_actions import (
    click,
    double_click,
    key_press,
    press_hotkey,
    right_click,
    screenshot,
    search_application,
    write_text,
)
from kgce.core import EnvironmentConfig

WINDOWS_ENV = EnvironmentConfig(
    name="windows",
    action_space=[
        click,
        key_press,
        write_text,
        press_hotkey,
        search_application,
        right_click,
        double_click,
    ],
    observation_space=[screenshot],
    description="""A Windows 11 desktop operating system. The interface \
displays a current screenshot at each step and primarily supports interaction \
via mouse and keyboard. You must use searching functionality to open any \
application in the system. This device includes system-related applications \
including Windows Terminal, Onenote, and Settings. It also features \
Google Chrome as the web browser, and the Microsoft Office Word, \
Excel, and PowerPoint. For communication, qq is available. The Google account \
is pre-logged in on Google Chrome, synchronized with the same account used in \
the Android environment.""",
)