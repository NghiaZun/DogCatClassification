const formData = require('form-data');
const fetch = require('node-fetch');
const { Buffer } = require('buffer');

exports.handler = async (event, context) => {
    if (event.httpMethod !== 'POST') {
        return {
            statusCode: 405,
            body: JSON.stringify({ error: 'Method Not Allowed' })
        };
    }

    const form = new formData();
    const imageBuffer = Buffer.from(event.body, 'base64');
    form.append('image', imageBuffer, { filename: 'image.jpg' });

    try {
        const response = await fetch('https://yourbackendurl.onrender.com/predict', {
            method: 'POST',
            body: form,
        });
        const data = await response.json();
        return {
            statusCode: 200,
            body: JSON.stringify({ label: data.label })
        };
    } catch (error) {
        return {
            statusCode: 500,
            body: JSON.stringify({ error: 'Internal Server Error' })
        };
    }
};
