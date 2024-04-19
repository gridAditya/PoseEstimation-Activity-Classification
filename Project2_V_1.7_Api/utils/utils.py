import jwt, cv2, base64, math
import numpy as np
from fastapi import HTTPException
from sklearn.cluster import KMeans
from collections import Counter
import json

SECRET_KEY="magatsuKami"
ALGORITHM="HS256"

def verify_auth_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

def process_images(image: np.ndarray, pose_image: np.ndarray):
    image_encoded = base64.b64encode(cv2.imencode('.png', image)[1]).decode()
    pose_image_encoded = base64.b64encode(cv2.imencode('.png', pose_image)[1]).decode()
    return [image_encoded, pose_image_encoded]


def run_kmeans(num_clusters, image):
    # Resize the image to (640, 640)
    image = cv2.resize(image, (640, 640))
    image = image.reshape((-1, 3))
    clusters = KMeans(num_clusters, random_state=42)
    clusters.fit(image)
    colors = clusters.cluster_centers_.astype(int).tolist()
    
    labels = clusters.labels_
    cluster_counts = Counter(labels)
    cluster_counts_final = dict()
    for key in cluster_counts.keys():
        cluster_counts_final[int(key)] = int(cluster_counts[key])
    
    # Let's save to a file the extracted data
    # Save the centroid
    try:
        with open('centroid.json', 'w') as f:
            json.dump(colors, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Could not save the centroids"})
    
    try:
        with open('counts.json', 'w') as f:
            json.dump(cluster_counts_final, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Could not save the cluster counts"})
    return 1

def load_json_data(path, error_message):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": error_message})    

def sum_of_squares(lst):
    return sum(x ** 2 for x in lst)


def calc_percent_of_major_shades():
    centroid, counts = load_json_data("centroid.json", "Could not load the centroids"), load_json_data("counts.json", "Could not load the cluster counts")
    
    # Create a centroid->cluster_no hash
    cluster_no = dict() # mapping from centroid->cluster_no
    cluster_centroid_dict = dict() # mapping from cluster_no->centroid

    for i in range(len(centroid)):
        key  = str(centroid[i][0]) + "_" + str(centroid[i][1]) + "_" + str(centroid[i][2])
        cluster_no[key] = i
        cluster_centroid_dict[i] = [centroid[i][0], centroid[i][1], centroid[i][2]]
    
    # Lets sort the centroid
    centroid = sorted(centroid, key=sum_of_squares)

    # We create the groups such that the distance in color space between min and max <= 10% of entire color space(255 * sqrt(3))
    group_hash = dict()
    grouped_centroids = []
    for i in range(len(centroid)):
        if i in group_hash:
            continue

        # Find the grp key and the original index before sorting
        grp_key = str(centroid[i][0]) + "_" + str(centroid[i][1]) + "_" + str(centroid[i][2])
        grp = [cluster_no[grp_key]]
        
        # Mark this position as visited
        group_hash[i] = True
        x1, y1, z1 = centroid[i][0], centroid[i][1], centroid[i][2]

        for j in range(i + 1, len(centroid)):
            # Calculate the distance between them <= 10% of the entire color space
            x2, y2, z2 = centroid[j][0], centroid[j][1], centroid[j][2]
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
            if (dist < 0.1 * 255 * math.sqrt(3)):
                # Find the grp key and the original index before sorting
                grp_key = str(centroid[j][0]) + "_" + str(centroid[j][1]) + "_" + str(centroid[j][2])
                grp.append(cluster_no[grp_key])
                
                # Marks this position as visited
                group_hash[j] = True
        grouped_centroids.append(grp)
    
    # Now let's calculate the area under each group
    # First caculate total points
    total_points = 0
    for key in counts.keys():
        total_points += counts[key]

    # Now we calculate the percentage share of each grp
    grp_area = [] # [percent, centroid]
    for i in range(len(grouped_centroids)):
        grp_share = 0
        grp_centroid = [0, 0, 0]
        for j in range(len(grouped_centroids[i])):
            # Find the share in image color space
            grp_member = grouped_centroids[i][j]
            share = (counts[str(grp_member)] / total_points) * 100
            grp_share += share
            
            # Find the current centroid
            curr_centroid = cluster_centroid_dict[grp_member]
            grp_centroid[0] += curr_centroid[0]
            grp_centroid[1] += curr_centroid[1]
            grp_centroid[2] += curr_centroid[2]
        grp_centroid[0] = int(grp_centroid[0] / len(grouped_centroids[i]))
        grp_centroid[1] = int(grp_centroid[1] / len(grouped_centroids[i]))
        grp_centroid[2] = int(grp_centroid[2] / len(grouped_centroids[i]))

        grp_area.append([grp_share, grp_centroid])
    grp_area.sort(reverse=True)

    if len(grp_area) < 2:
        return False, [[grp_area[0][0], grp_area[0][1]]]
    top_two_grp_sum = grp_area[0][0] + grp_area[1][0]
    print(top_two_grp_sum)

    if top_two_grp_sum >= 75:
        return False, [[grp_area[0][0], grp_area[0][1]], [grp_area[1][0], grp_area[1][1]]]
    return True, []
