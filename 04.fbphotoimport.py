import facebook,os,shutil,requests

token = "EAAIKcWqceBwBAAqvEcfq4Uq0p4HAqLQZAYUOuKFKv1eIGJUyXERSpriDFHEQOaXOWrs1HEGK3bjzHkBmuZB8nhLJUzkeNXeBfCr42P0pYrNtavgxu19ZA9p13nEH5jAGROkrx5y9gUMVXDpzrzIsF0aushQhgI2SyljQ5Qr6FaVTNY4kQVW7dPJm3iQUBUZD"

graph = facebook.GraphAPI(access_token=token, version='2.7')

pages = graph.get_connections(id='me', connection_name='photos?fields=images')
photos = pages['data']

if not os.path.exists("fb-photos"):
    os.mkdir("fb-photos")

for photo in photos:
    imageList = photo['images']
    for img in imageList:
        if img['width'] > 1000:
            fileName = img['source'].split('/')[-1].split('?')[0]
            f = open('fb-photos/'+fileName, 'wb')
            pic = requests.get(img['source'], stream=True)
            shutil.copyfileobj(pic.raw, f)
            f.close()