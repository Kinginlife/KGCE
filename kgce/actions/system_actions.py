#
# import subprocess
# from time import sleep
#
# from kgce.core.decorators import action
#
#
# @action
# def delay(time: float) -> None:
#     sleep(time)
#
#
# @action
# def run_bash_command(command: str) -> str:
#     """
#     Run a command using bash shell. You can use this command to open any application by
#     their name.
#
#     Args:
#         command: The commmand to be run.
#
#     Return:
#         stdout and stderr
#     """
#     p = subprocess.run(["bash", command], capture_output=True)
#     return f'stdout: "{p.stdout}"\nstderr: "{p.stderr}"'
import subprocess
from time import sleep
import sys
from kgce.core.decorators import action


@action
def delay(time: float) -> None:
    """
    Delay for the specified time.

    Args:
        time: The time to delay, in seconds.
    """
    sleep(time)


@action
def run_command(command: str) -> str:
    """
    Run a command using the system shell. This function is compatible with both
    Windows (PowerShell) and Unix-like systems (bash).

    Args:
        command: The command to be run.

    Returns:
        A string containing the stdout and stderr of the command.
    """
    # Determine the shell based on the operating system
    if sys.platform == "win32":
        # Use PowerShell on Windows
        shell = "powershell"
        command = ["-Command", command]
    else:
        # Use bash on Unix-like systems
        shell = "bash"
        command = ["-c", command]

    # Run the command
    p = subprocess.run([shell, *command], capture_output=True, text=True)

    # Return stdout and stderr
    return f'stdout: "{p.stdout}"\nstderr: "{p.stderr}"'