# Old-Projects
A repo containing some projects from over the last year.

We have a selection of projects I've worked on over the last year in various states of completion. These were originally made just for me so they need to be polished up slightly before they are fit for public consumption.

## PixelArt ## 
- Often on social media people present art which has been assembled by combining a number of discrete objects. Examples include art formed from Rubix cubes, dice or sticky notes. I thought, given some finite discrete palette of colours and a target image, can we write a program which produces an image using these colours which is 'near' to this target. In this way we can use this as a colour-by-numbers to help us produce similar art. Below is a picture of a dog made out of coloured thumbtacks made using this method. (Approximately August 2021). At high zoom the pixelated image is highly unconvincing as a dog. However with increased distance the likeness improves a lot (see bottom right).

Example: Below is an example using a picture of a dog. We resize this to 28 by 28 pixels and recreate this using thumbtacks. This is a simply proof of concept and results will look better for higher resolutions and, of course, broader colour palettes. 

![alt text](https://github.com/WToner/Old-Projects/blob/main/ThreeDogsPixels.png?raw=true)

- This project needs to be worked on much more when time permits. The current primary strategy involves 'rounding' each pixel to the nearest palette colour. This is a very poor methodology. Account needs to be taken of the average colour of the resulting images of patches of various sizes.

## Crypto ## 
- A project from around Jan 2021 for predicting crypto-currency prices. Historical prices were scraped from various exchanges. Using these one can train simple regression models in an attempt to uncover underlying patterns. 

## Scraper ##
- Originally designed to scrape hotel prices from Booking.com; shelved over concerns about legality. (Approximately November 2021)

## ScreenCatch ##
- There are number of channels on TikTok which garner viewers by scratching gift cards live on air. The money goes to the first person to claim the card. This project detects and saves text to clipboard automatically. It uses as it's foundation the open source vision software Tesseract OCR. (January 2021)

## DeepFace ## 
- I thought it would be neat to make a program which converts a source video of a person doing an action, to a video of Donald Trump carrying out this action. The method used is taken from the paper "Everyone Can Dance". We use an opensource landmark detector (https://github.com/cunjian/pytorch_face_landmark) and apply it to each frame of a video of Donald Trump. From this we then train a neural network to reconstruct the original image. Having done this we can move to the source video. This is likewise divided into frames which are then converted into landmarks. These are fed into the pre-learnt network to obtain the desired video. Care must be taken to ensure that the source frames are within the support of the trained network and it is here where issues can manifest. Additional problems result from a lack of smooth temporal consistency unless this is included as a loss.
-As mentioned before, the landmarks are obtained using code from the following repository https://github.com/cunjian/pytorch_face_landmark None of the code for obtaining face landmarks is mine, only the code which builds upon this for creating videos.





