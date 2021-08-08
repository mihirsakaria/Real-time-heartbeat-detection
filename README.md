# Average heartbeat detection

For doit.py
Detects heartbeat from the video of the face.
Based on Eulerian Video Magnification

Input conditions:
The input should consist of a video file of the person's face, nearly 30-40 seconds for best results.During the video, the person should be steady for atleast 30seconds. Headmovements and other unnecessary movements should be avoided.

The code is for Eulerian Video magnification, specifically used for printing the heartbeat of a person measured from the video of the face. Whenever the heart pumps blood, the blood goes into the cells and there is a change in color intensity which is invisible to the naked eye. So Eulerian Video magnification helps in detecting the intensity change and amplifying it. The frequencies changes are measured overtime and thus heartbeat is calculated with that.

The best region to check the output is the forehead region. Its where the intensity change is clearly visible. So it is the region of interest.
The green band is observed to give best intensity changes. So the green band is extracted from the region of interest.

The above approach gives us average heartrate over the entire interval of the video.

# Real time heartrate and Spo2 level detection

Eulerian Video magnification is helpful to one extent, but not a good method to find heartrate in real time
The approach and be extended to calculate the frequency change of pixels for each second. As the fps speed is 30 this means that we can calculate the change in pixels for these 30 frames and thus that would be the heartrate for each second.
Also using various correlation formulas it was found that the heart rate and blood oxygen levels are inversly correlated.
So using that correlation we can determine the blood oxygen levels using the heartrates.
