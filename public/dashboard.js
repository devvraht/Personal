// // Redirect to login if not logged in
// const token = localStorage.getItem('token');
// if (!token) {
//   window.location.href = 'login.html';
// }

const videoContainer = document.getElementById("videoContainer");
const mapContainer = document.getElementById("mapContainer");

const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
videoContainer.appendChild(canvas);

const img = new Image();

const mapDiv = document.createElement("div");
mapDiv.style.width = "100%";
mapDiv.style.height = "100%";
mapContainer.appendChild(mapDiv);

const map = L.map(mapDiv).setView([0, 0], 2);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19
}).addTo(map);

const marker = L.marker([0, 0]).addTo(map);

let latestFrame = null;
let latestLat = 0;
let latestLng = 0;

let isSwapped = false;

const ws = new WebSocket("wss://15.206.149.220/ws");

ws.onmessage = (event) => {
  const parts = event.data.split("|", 2);
  if (parts.length !== 2) return;

  const [latlng, base64] = parts;
  const coords = latlng.split(",");

  if (coords.length !== 2) return;

  latestLat = parseFloat(coords[0]);
  latestLng = parseFloat(coords[1]);
  latestFrame = base64;

  marker.setLatLng([latestLat, latestLng]);
  map.setView([latestLat, latestLng], 15, { animate: false });
};

function renderLoop() {
  if (latestFrame) {
    img.src = "data:image/jpeg;base64," + latestFrame;

    img.onload = () => {
      if (canvas.width !== img.width) {
        canvas.width = img.width;
        canvas.height = img.height;
      }
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
  }
  requestAnimationFrame(renderLoop);
}
renderLoop();

mapContainer.addEventListener("click", () => {

  if (!isSwapped) {
    videoContainer.appendChild(mapDiv);

    mapContainer.appendChild(canvas);

    isSwapped = true;
    const div = document.getElementById('mapContainer');
    div.style.zIndex = 10000;
  } else {
    videoContainer.appendChild(canvas);
    mapContainer.appendChild(mapDiv);

    isSwapped = false;
    const div = document.getElementById('mapContainer');
    div.style.zIndex = 0;
  }

  setTimeout(() => {
    map.invalidateSize();
  }, 300);
});

// Redirect to login if no JWT
// const token = localStorage.getItem("token");
// if (!token) {
//   window.location.href = "login.html";
// }

// Logout
document.getElementById("logoutBtn").addEventListener("click", () => {
  localStorage.removeItem("token");
  window.location.href = "login.html";
});


// Logout button
const logoutBtn = document.getElementById("logoutBtn");

logoutBtn.addEventListener("click", async () => {
  try {
    // Optional: inform server to invalidate token if you implement token blacklisting
    // await fetch('/api/logout', {
    //   method: 'POST',
    //   headers: { 'Authorization': 'Bearer ' + token }
    // });

    // Remove JWT from localStorage
    localStorage.removeItem("token");

    // Redirect to login page
    window.location.href = "login.html";
  } catch (err) {
    console.error("Logout failed:", err);
  }
});