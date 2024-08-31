import requests
import tarfile
import sys
import os


def get_dataset(url):
    """Download dataset from the given url
    Args
        url: URL for the dataset
    """    
    file_name = url.split('/')[-1]

    # Download file and print the progress
    with open(file_name, "wb") as f:
        print("Downloading {}".format(file_name))
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()
    
    # Extract the tar file
    tar = tarfile.open(file_name)
    tar.extractall()
    tar.close()
    os.remove(file_name)