const loginForm = document.getElementById("loginForm");
const errorMsg = document.getElementById("error");

loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value.trim();

  try {
    const response = await fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });

    const data = await response.json();

    if (data.success) {
      window.location.href = "dashboard.html";
    } else {
      errorMsg.textContent = "Invalid username or password!";
    }
  } catch (err) {
    console.error(err);
    errorMsg.textContent = "Server error. Try again later.";
  }
});