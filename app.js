// app.js
document.getElementById('uploadButton').addEventListener('click', () => {
    document.getElementById('imageInput').click();
});

document.getElementById('imageInput').addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = () => {
            document.getElementById('selectedImage').src = reader.result;
            document.getElementById('selectedImage').style.display = 'block';
            uploadImage(file);
        };
        reader.readAsDataURL(file);
    }
});

async function uploadImage(file) {
    document.getElementById('loading').style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('https://master--sheridancapstone.netlify.app/.netlify/functions/app/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        document.getElementById('artwork').textContent = `Name: ${data.artwork}`;
        document.getElementById('artist').textContent = `Artist: ${data.artist}`;
        document.getElementById('date').textContent = `Date: ${data.date}`;
        document.getElementById('style').textContent = `Style: ${data.style}`;

        document.getElementById('prediction').style.display = 'block';
    } catch (error) {
        console.error('Error uploading image:', error);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}
