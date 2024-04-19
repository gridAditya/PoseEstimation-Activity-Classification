const button = document.getElementById("submit_button")


button.addEventListener("click", async function(event){
    event.preventDefault()
    const username = document.getElementById("username").value
    const email = document.getElementById("email").value
    const password = document.getElementById("password").value
    const confirm_password = document.getElementById("confirm_password").value
    
    if (username.length === 0 || email.length === 0 || password.length === 0 || confirm_password.length === 0){
        alert("[-] All feilds are required!")
        return
    }
    if (password !== confirm_password){
        alert("[-] Incorrect password!")
        return
    }
    try{
        const formData = new FormData();
        formData.append("username", username);
        formData.append("email", email);
        formData.append("password", password);
        formData.append("confirm_password", confirm_password);

        const response = await fetch("http://127.0.0.1:8000/signup", {
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
        }
    }catch (error) {
        console.error("[-] Server_Error:", error);
    }
})

