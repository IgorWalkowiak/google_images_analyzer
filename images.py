import cv2 as cv
from google_images_download import google_images_download
import os

CHROME_DRIVER_PATH = "D:\chromedriver_win32\chromedriver.exe"

class Images:
    def __init__(self, images_dir, image_size_x, image_size_y, limit):
        self.limit = limit
        self.images_dir = images_dir
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y


    def downloadImages(self, query):
        response = google_images_download.googleimagesdownload()
        arguments = {"keywords": query,
                     "format": "jpg",
                     "limit": self.limit,
                     "print_urls": True,
                     "size": "medium",
                     "aspect_ratio": "square",
                     "chromedriver": CHROME_DRIVER_PATH,
                     "count_offset": 1,
                     "output_directory": self.images_dir
                     }

        try:
            response.download(arguments)
        except FileNotFoundError:
            try:
                response.download(arguments)
            except:
                print("ERROR!")
        self._parseImages(query)

    def _parseImages(self, query):
        fileNames = os.listdir(self.images_dir + '/' + query)
        for fileName in fileNames:
            fileDir = self.images_dir + '/' + query + '/' + fileName
            try:
                src = cv.imread(fileDir, cv.IMREAD_UNCHANGED)
                dsize = (self.image_size_x, self.image_size_y)
                output = cv.resize(src, dsize)
                cv.imwrite(fileDir, output)
            except Exception as e:
                print(str(e))



