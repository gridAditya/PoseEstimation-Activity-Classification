const logout = document.getElementById("logout")
const logoutAll = document.getElementById("logoutAll")

// Get the modal
const modal = document.getElementById("myModal");
const error_modal = document.getElementById("error-modal")
const wait_modal = document.getElementById("wait-modal")

const close_button = document.getElementsByClassName("close")[0];
const error_close_button = document.getElementsByClassName("close")[1];
const label = document.getElementById("label")

// Get the image tags
const original_image = document.getElementById("original_image");
const pose_image = document.getElementById("pose_image")

function getRandomNumber(low, high) {
    return Math.floor(Math.random() * (high - low + 1)) + low;
}

document.getElementById("uploadButton").addEventListener("click", async function(event)
{
    event.preventDefault(); // Prevent the form from submitting normally
    const low = 10000
    const high = 99999

    const formData = new FormData();
    const imageDescription = document.getElementById("imageDescription").value;
    const imageFile = document.getElementById("imageUpload").files[0];
    
    formData.append("imageDescription", imageDescription);
    formData.append("imageUpload", imageFile);
    randomThrow = getRandomNumber(low, high)
    console.log(randomThrow)
    
    try{
        token = localStorage.getItem("token")
        wait_modal.style.display = "block"
        const response = await fetch(`http://127.0.0.1:8000/evaluate_pose/${randomThrow}`, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`
            },
            body: formData
        });
        wait_modal.style.display = "none"
        if (response.ok){
            const data = await response.json()
            if (data.error){
                const heading = document.getElementById("error-heading")
                heading.textContent = data.error.detail.error
                
                // Remove any previous elements that were appended
                const parent_block = document.getElementById("shade-block")
                while (parent_block.firstChild) {
                    parent_block.removeChild(parent_block.firstChild);
                }

                // If this is a shade error show the shade graph
                if ("type" in data.error.detail){
                    centroids = data.error.detail.centroid
                    // console.log(centroids)
                    for (let i = 0; i < centroids.length; i++){
                        const shade = document.createElement("span")
                        shade.classList.add("shade")
                        shade.style.width = centroids[i][0].toString() * 0.01 * 400 + "px"
                        shade.textContent = (parseInt(centroids[i][0])).toString() + "%"

                        shade.style.color = "white"
                        shade.style.textAlign = "center"
                        shade.style.height = "20px"
                        shade.style.backgroundColor = `rgb(${centroids[i][1][2]}, ${centroids[i][1][1]}, ${centroids[i][1][0]})`
                        parent_block.appendChild(shade)
                    }
                }
                
                // Display the heading
                error_modal.style.display = "block"
            }else{
                // Decode base64 strings to binary image data
                if (!("error" in data)){
                    const imageSrc = 'data:image/png;base64,' + data.image;
                    const poseImageSrc = 'data:image/png;base64,' + data.pose_image;
                    const image_label = data.label
                    
                    // Insert the images
                    original_image.src = imageSrc
                    pose_image.src = poseImageSrc
                    label.textContent = image_label.toUpperCase()

                    // Open the modal
                    modal.style.display = "block";
                }
                else{
                    console.log(data)
                }
            }
        }else {
            const error = await response.json()
            console.error("[-] Error:", error);
        }
    }catch (error) {
        console.error("[-] Server_Error:", error);
    }
});

logout.addEventListener("click", async function(event){
    try{
        token = localStorage.getItem("token")

        const response = await fetch("http://127.0.0.1:8000/logout", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            }
        })

        if (response.ok){
            const data = await response.json()
            console.log("[+] Logged out successfully...")
            console.log(data)
            // Erase the token from localStorage
            localStorage.removeItem("token")
            
            // Redirect to homePage
            window.location.href = "http://127.0.0.1:8000/"
        }
    }catch(error){
        console.log("[-] Unable to logout....")
        console.log(error)
    }
})

logoutAll.addEventListener("click", async function(event){
    try{
        token = localStorage.getItem("token")

        const response = await fetch("http://127.0.0.1:8000/logoutAll", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            }
        })

        if (response.ok){
            const data = await response.json()
            console.log("[+] Logged out successfully from all sessions...")
            console.log(data)
            // Erase the token from localStorage
            localStorage.removeItem("token")
            
            // Redirect to homePage
            window.location.href = "http://127.0.0.1:8000/"
        }
    }catch(error){
        console.log("[-] Unable to logout....")
        console.log(error)
    }
})

close_button.onclick = function() {
    modal.style.display = "none";
}

error_close_button.onclick = function(){
    error_modal.style.display = "none"
}

window.onclick = function(event) {
    if (event.target == modal) {
      modal.style.display = "none";
    }
}