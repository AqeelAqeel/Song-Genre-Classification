import requests
import os

print(os.getcwd())

genres = ['hip-hop', 'classical', 'country', 'electronic', 'metal']

n_pages = 40
start_page = 1

def write_to_file(url, genre, i):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open("{}/{}/{:04d}{}".format("data", genre, i,local_filename), 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian


# go through number of pages for each genre from parent website 
for genre in genres:
    
    counter =  1

    # if genre != 'electronic':
    #     counter = 1
    #     start_page = 1
    
    for i in range(start_page, n_pages+1):
        if i == 1:
            url = f'https://elements.envato.com/audio/genre-{genre}'
        else:
            url = f'https://elements.envato.com/audio/genre-{genre}/pg-{i}'
        new_url = 'http://api.scraperapi.com?api_key=be11db095d4ce5fbffe22179b20cbcc3&url=' + url 
        
        # get HTML source code of parent website as a string using .text
        response = requests.get(new_url).text

        # based on string of HTML source code, split on the pattern of desired URL links per page (which come after 'src')
        response_split = response.split("src=")

        for split in response_split:
            try:
                # look to see what index of string url contains 'mp3' to identify split location from html url link 
                idx = split.index('mp3')
                mp3_url = split[idx-78:idx+3]
                print(f"Currently on page {i}!")
                #make sure the url contains the audio file
                if mp3_url.startswith('https:'):
                    #save file to specified local drive
                    write_to_file(mp3_url, genre, counter)
                    counter += 1
            except:
                pass


'''
BASH TERMINAL COMMAND:

for each folder:
   go in folder   
   copy file to a 30 second version

while in terminal in subdirectory for each genre of musicData folder:

for i in *.mp3; do ffmpeg -i $i -ss 00:00:00 -to 00:00:30 -c copy shortened_$i  ;done

'''

