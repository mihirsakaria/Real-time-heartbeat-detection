# Real time heartbeat detection

Detects heartbeat from the video of the face.
Based on Eulerian Video Magnification

Input conditions:
The input should consist of a video file of the person's face, nearly 30-40 seconds for best results.During the video, the person should be steady for atleast 30seconds. Headmovements and other unnecessary movements should be avoided.

The code is for Eulerian Video magnification, specifically used for printing the heartbeat of a person measured from the video of the face. Whenever the heart pumps blood, the blood goes into the cells and there is a change in color intensity which is invisible to the naked eye. So Eulerian Video magnification helps in detecting the intensity change and amplifying it. The frequencies changes are measured overtime and thus heartbeat is calculated with that.

The best region to check the output is the forehead region. Its where the intensity change is clearly visible. So it is the region of interest.
The green band is observed to give best intensity changes. So the green band is extracted from the region of interest.
