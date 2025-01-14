import datetime

timestamp = 1667918015
# Convert the timestamp to a datetime object
dt_object = datetime.datetime.utcfromtimestamp(timestamp)

# Convert the datetime object to a human-readable format
human_readable_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')

print("Human-readable time:", human_readable_time)