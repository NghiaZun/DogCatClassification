const apiUrl = process.env.MODEL_API_URL;
const apiKey = process.env.MODEL_API_KEY;
const FormData = require('form-data');

exports.handler = async (event, context) => {
    if (event.httpMethod !== 'POST') {
        return {
            statusCode: 405,
            body: JSON.stringify({ error: 'Method Not Allowed' })
        };
    }

    if (!apiUrl || !apiKey) {
        return {
            statusCode: 500,
            body: JSON.stringify({ error: 'API configuration is missing' })
        };
    }

    try {
        const { image } = JSON.parse(event.body); // Expecting `image` in the request body
        const formData = new FormData();
        formData.append('file', image);

        const response = await fetch(`${apiUrl}/predict`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                // `fetch` will automatically set the correct Content-Type for FormData
            },
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`API responded with status ${response.status}`);
        }

        const data = await response.json();
        return {
            statusCode: 200,
            body: JSON.stringify({ label: data.label })
        };
    } catch (error) {
        return {
            statusCode: 500,
            body: JSON.stringify({ error: error.message || 'Internal Server Error' })
        };
    }
};