const apiUrl = process.env.MODEL_API_URL;
const apiKey = process.env.MODEL_API_KEY;

exports.handler = async (event, context) => {
    if (event.httpMethod !== 'POST') {
        return {
            statusCode: 405,
            body: JSON.stringify({ error: 'Method Not Allowed' })
        };
    }

    try {
        const response = await fetch(`${apiUrl}/predict`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`
            },
            body: formData,
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
