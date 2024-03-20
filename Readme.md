<h1>Pose Estimation and Activity Classification</h1>

<h2>How to structure the downloaded data</h2>
<p>After downloading the MPII Human Pose Dataset, it will have two folders 'mpii_human_pose_v1_u12_2' and 'images'.<br>
Rename the 'images' folder to 'original_images' and then place both the folders inside '/data/raw'.<br>
<br>&nbsp;
Also inside the '/data/processed' directory the structure should be like:<br>
<b>Structure of the processed data directory:</b><br>
root/data/processed<br>
|____ images<br>
|&emsp;&emsp;&emsp;|_______train<br>
|&emsp;&emsp;&emsp;|_______val<br>
|____ labels<br>
|                    |_______train<br>
|                    |_______val<br>
|____ images_pose<br>
|                     |_______train<br>
|                     |_______val<br>
|____ labels_pose<br>
|                     |_______train<br>
|                     |_______val<br>
|____ train_4_channel_info<br>
|<br>
|____ train_labels_pose_class<br>
|<br>
|____ val_4_channel_info<br>
|<br>
|____ val_labels_pose_class<br>
</p>

<h2>What we have done</h2>
<ul>
    <li>Used the images to estimates the pose of the humans in the images
    <li>Used the images + pose to estimate what kind of activity the humans are doing(there are 20 classes of possible activities)
</ul>

<h3>Activities/Class</h3>
class_names = {<br>
&nbsp;'sports': 0,<br>
&nbsp;'miscellaneous': 1,<br>
&nbsp;'home activities': 2,<br>
&nbsp;'occupation': 3,<br>
&nbsp;'fishing and hunting': 4,<br>
&nbsp;'home repair': 5,<br>
&nbsp;'conditioning exercise': 6,<br>
&nbsp;'lawn and garden': 7,<br>
&nbsp;'religious activities': 8,<br>
&nbsp;'music playing': 9,<br>
&nbsp;'inactivity quiet/light': 10,<br>
&nbsp;'water activities': 11,<br>
&nbsp;'running': 12,<br>
&nbsp;'winter activities': 13,<br>
&nbsp;'walking': 14,<br>
&nbsp;'dancing': 15,<br>
&nbsp;'bicycling': 16,<br>
&nbsp;'transportation': 17,<br>
&nbsp;'self care': 18,<br>
&nbsp;'volunteer activities': 19<br>
}
