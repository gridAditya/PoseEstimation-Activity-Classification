const button = document.getElementById("submit_button")
const error_modal = document.getElementById("error-modal")
const close_button = document.getElementsByClassName("close")[0];

button.addEventListener("click", async function(event){
    event.preventDefault()
    const email = document.getElementById("email").value
    const password = document.getElementById("password").value
    
    if (email.length === 0 || password.length === 0){
        alert("[-] All feilds are required!")
        return
    }
    try{
        const formData = new FormData();
        formData.append("email", email);
        formData.append("password", password);

        const response = await fetch("http://127.0.0.1:8000/login", {
            method: "POST",
            body: formData
        })
        if (response.ok){
            const data = await response.json()
            console.log("[+] Response", data)
            
            // Save the token to localStorage
            localStorage.setItem("token", data.token)
            
            // Change the location to "evaluate_pose.html"
            window.location.href = "http://127.0.0.1:8000/evaluate_pose.html"
        }else {
            console.error("[-] Error:", response.statusText);
            const heading = document.getElementById("error-heading")
            heading.textContent = response.statusText

            // Display the heading
            error_modal.style.display = "block"

            // Clear the current localStorage
            localStorage.removeItem("token")
        }
    }catch (error) {
        console.log('s')
        console.error("[-] Server_Error:", error);
    }
})

close_button.onclick = function() {
    error_modal.style.display = "none";
}
