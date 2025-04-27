const fetch = require('node-fetch');

exports.handler = async (event, context) => {
  // URL của Cloud Run service
  const cloudRunUrl = "https://<your-service-name>-<random-id>.run.app/predict";
  
  // Lấy hình ảnh từ yêu cầu
  const formData = new FormData();
  formData.append('image', event.body);  // Giả sử hình ảnh được gửi từ yêu cầu POST

  try {
    const response = await fetch(cloudRunUrl, {
      method: 'POST',
      body: formData,
      headers: {
        'Authorization': `Bearer <your-api-key-if-needed>`,
      },
    });

    const data = await response.json();
    
    return {
      statusCode: 200,
      body: JSON.stringify(data),
    };
  } catch (error) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message }),
    };
  }
};
