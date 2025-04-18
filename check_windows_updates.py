import subprocess
import datetime

def get_installed_updates():
    """
    Retrieves a list of installed Windows updates for the current month using PowerShell.
    """
    try:
        # Get the current year and month
        now = datetime.datetime.now()
        year = now.year
        month = now.month

        # Construct the PowerShell command to get updates installed in the current month
        ps_command = f"""
        $Updates = Get-HotFix | Where-Object {{
            $_.InstalledOn -like '{month:02d}/*/{year}'
        }}
        $Updates | ForEach-Object {{
            Write-Output $($_.HotFixID + " - " + $_.InstalledOn)
        }}
        """

        # Execute the PowerShell command
        result = subprocess.run(["powershell", "-Command", ps_command], capture_output=True, text=True)

        # Check for errors
        if result.returncode != 0:
            print(f"Error executing PowerShell command: {result.stderr}")
            return []

        # Parse the output
        updates = result.stdout.strip().splitlines()
        return updates

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    updates = get_installed_updates()

    if updates:
        print("Windows updates installed this month:")
        for update in updates:
            print(update)
    else:
        print("No Windows updates installed this month.")