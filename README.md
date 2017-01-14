# Licence -Plate Check

## Goal:
The goal of this project is to create a system capable recognizing licence plates
in a image and check in a government server for its status (e.g check for car theft)

## Methodology:
...

### 1) Localize the licence plate
The first thing to do is to localize the licence plate in the image. We do that using
the Canny edge detector. After applying the edge detector, we look for contours in
the image and then fit rectangles to these contours. The rectangles with appropriate
area and aspect ratio as considered licence plates.

### 3) Segment digits
The Brazilian licence plate has a pattern: 
'letter' 'letter' 'letter' '-' 'number' 'number' 'number' 'number'
The background is the grey  and the symbols are black (for the standard licence plate).
First we threshold the image so the symbols are white and the background is black.
The regions between symbols are going to be all black. We can identify these
regions by making vertical sums over the image. The between symbol regions
will have small values for teh vertical sums (local minimuns in the vertical sum function).
If we crop our image between these local minimas, we have our symbols segmented.

### 4) Recognize each symbol
To recognize each symbol, we will train a feed-forward neural network with program
generated symbols.
The generated symbols will have random rotation, random amount of salt-and-pepper
noise and random thickness.

### 5) Check the licence plate status in a server
The Brazilian government has a website for public consultation
on licence plate status (https://www.sinesp.gov.br/). After getting the licence
plate the system will check the plate status in this website.


### 6) Raspberry PI implementation
After testing our system in recorded videos from parking lots, the next step is
to implent the system in real time. The system will be implemented using a 
Raspberry PI 3 and a kinect sensor camera.

