from crontab import CronTab
import os

# Define the Python script you want to run
script_path = "./scripts/model_rebuild_scheduler.py"
python_path = "/usr/bin/python3"  # Change this to the path to your Python executable, if necessary

# Create a CronTab object for the current user
cron = CronTab(user=True)

# Define the cron job command
command = f"{python_path} {script_path}"

# Create a new cron job to run the script every 30 minutes
job = cron.new(command=command)

# Set the job to run every 5 minutes
job.minute.every(5)

# Save the new cron job
cron.write()

print(f"Cron job created to run the script every 30 minutes: {command}")
