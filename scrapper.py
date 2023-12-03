import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import csv
import datetime


cred = credentials.Certificate('serms-app-firebase-adminsdk-tjue0-50f8c4ff82.json')
firebase_admin.initialize_app(cred)

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

bucket_name = 'serms-app.appspot.com'
urls = get_storage_urls(bucket_name)

for url in urls:
    print(url)


def save_urls_to_csv(urls, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['URLs'])
        for url in urls:
            writer.writerow([url])

# Retrieve URLs from Firebase Storage
bucket_name = 'serms-app.appspot.com'
urls = get_storage_urls(bucket_name)

# Save URLs to a CSV file
file_path = 'urls.csv'  # Specify the path where you want to save the CSV file
save_urls_to_csv(urls, file_path)