# %%
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import csv
import datetime


# %%
# Initialize Firebase Admin SDK
cred = credentials.Certificate('serms-app-firebase-adminsdk-tjue0-6280b0f23a.json')
firebase_admin.initialize_app(cred)

# %%
def get_storage_urls(bucket_name):
    urls = []
    bucket = storage.bucket(bucket_name)
    blobs = bucket.list_blobs()

    for blob in blobs:
        url = blob.generate_signed_url(
            version='v4',  # Modify to the desired URL format/version
            expiration=datetime.timedelta(days=1),  # Set the URL expiration as needed
            method='GET'
        )
        urls.append(url)

    return urls


# %%
bucket_name = 'serms-app.appspot.com'
urls = get_storage_urls(bucket_name)

for url in urls:
    print(url)


# %%


# %%
def save_urls_to_csv(urls, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['URLs'])
        for url in urls:
            writer.writerow([url])

# Retrieve URLs from Firebase Storage
bucket_name = 'your-firebase-storage-bucket-name'
urls = get_storage_urls(bucket_name)

# Save URLs to a CSV file
file_path = 'urls.csv'  # Specify the path where you want to save the CSV file
save_urls_to_csv(urls, file_path)


# %%


# %%
def merge_urls_with_csv(csv_file_path, urls):
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # Add URL column header to the CSV file
    header = rows[0].keys()
    header = list(header) + ['URL']
    for row, url in zip(rows, urls):
        row['URL'] = url

    # Save the merged data to a new CSV file
    merged_csv_file_path = 'merged_song_details.csv'  # Specify the path for the merged CSV file
    with open(merged_csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Merged data saved to: {merged_csv_file_path}")


# Retrieve URLs from Firebase Storage
bucket_name = 'serms-app.appspot.com'
urls = get_storage_urls(bucket_name)

# Merge URLs with the song_details.csv file
csv_file_path = 'songdetails.csv'  # Specify the path to the song_details.csv file
merge_urls_with_csv(csv_file_path, urls)



