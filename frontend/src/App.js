import React, { useState } from 'react';

const App = () => {
  // State variables for document URL, questions input, and response
  // Changed initial state to empty strings for placeholders
  const [documentUrl, setDocumentUrl] = useState('');
  const [questionsInput, setQuestionsInput] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Function to handle form submission when the "Retrieve Answers" button is clicked
  const handleSubmit = async () => {
    setLoading(true); // Set loading state to true
    setError(null);    // Clear any previous errors
    setResponse(null); // Clear any previous responses

    // Split the questions input by new line to get an array of individual questions
    // Filter out any empty lines that might result from extra newlines
    const questions = questionsInput.split('\n').filter(q => q.trim() !== '');

    // Construct the payload to be sent to your backend API.
    // The 'documents' field will now contain the URL.
    const payload = {
      documents: documentUrl, // This sends the URL to your backend
      questions: questions,
    };

    try {
      // Define the URL for your FastAPI backend endpoint.
      // Ensure your backend is running on http://localhost:8000
      const apiUrl = "http://localhost:8000/api/v1/hackrx/run"; // Base URL from image_f0a99b.png

      // Make the API call to your backend using fetch.
      const apiResponse = await fetch(apiUrl, {
        method: 'POST', // Use POST method as per API documentation
        headers: {
          'Content-Type': 'application/json', // Specify content type as JSON
          'Accept': 'application/json',       // Expect JSON response
          // Include the Authorization header as per image_f0a99b.png
          'Authorization': 'Bearer 5ba298f10582e591e01dd5a437f580a8da1354f64898b2042fc74b9e0968f9d1',
        },
        body: JSON.stringify(payload) // Convert the payload object to a JSON string
      });

      // Check if the API response was successful (status code 2xx)
      if (!apiResponse.ok) {
        // If not successful, try to parse error details from the response
        const errorData = await apiResponse.json();
        // Throw an error with a specific message or a generic one
        throw new Error(errorData.detail || 'Failed to retrieve answers from backend.');
      }

      // Parse the successful response from the backend as JSON
      const result = await apiResponse.json();
      setResponse(result); // Update the response state with the received data

    } catch (err) {
      // Catch any errors during the fetch operation or response processing
      setError('Failed to fetch response from backend: ' + err.message);
      console.error("Error during API call:", err);
    } finally {
      setLoading(false); // Always set loading state to false after the operation completes
    }
  };

  // Function to copy answers to clipboard
  const copyToClipboard = () => {
    if (response && response.answers) {
      const answersText = response.answers.map((answer, index) => `${index + 1}. ${answer}`).join('\n');
      // Use document.execCommand('copy') for broader iframe compatibility
      const textArea = document.createElement("textarea");
      textArea.value = answersText;
      document.body.appendChild(textArea);
      textArea.select();
      try {
        document.execCommand('copy');
        alert('Answers copied to clipboard!'); // Use a custom message box in a real app
      } catch (err) {
        console.error('Failed to copy text: ', err);
        alert('Failed to copy answers.');
      }
      document.body.removeChild(textArea);
    }
  };


  return (
    // Main container div for the entire application, applying basic styling
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4 font-inter">
      {/* Tailwind CSS CDN for quick styling. For a local project, consider installing Tailwind properly. */}
      <script src="https://cdn.tailwindcss.com"></script>
      {/* Google Fonts link for the 'Inter' font */}
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
      {/* Removed the problematic Font Awesome CDN integrity attribute */}
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" />

      {/* Inline style block for the custom font-family */}
      <style>
        {`
          .font-inter {
            font-family: 'Inter', sans-serif;
          }
        `}
      </style>

      {/* Main content card with white background, shadow, and rounded corners */}
      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-3xl">
        {/* Application title */}
        <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          LLM-Powered Query Retriever
        </h1>

        {/* Document URL Input field */}
        <div className="mb-4">
          <label htmlFor="documentUrl" className="block text-gray-700 text-sm font-medium mb-2">
            Document URL (PDF, DOCX, or Text URL)
          </label>
          <input
            type="text"
            id="documentUrl"
            className="shadow-sm appearance-none border rounded-lg w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200 ease-in-out"
            value={documentUrl} // Binds input value to documentUrl state
            onChange={(e) => setDocumentUrl(e.target.value)} // Updates state on input change
            placeholder="e.g., https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A3AZ&se=2025-07-04T09%3A11%3A3AZ&sr=b&sp=r&sig=YOUR_SIGNATURE_HERE" // Placeholder text
            aria-label="Document URL"
            disabled={loading} // Disable input when loading
          />
          <p className="text-xs text-gray-500 mt-1">
            Ensure the URL is publicly accessible for your backend to fetch and process it.
          </p>
        </div>

        {/* Questions Input textarea */}
        <div className="mb-6">
          <label htmlFor="questions" className="block text-gray-700 text-sm font-medium mb-2">
            Questions (one per line)
          </label>
          <textarea
            id="questions"
            className="shadow-sm appearance-none border rounded-lg w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200 ease-in-out h-48 resize-y"
            value={questionsInput} // Binds textarea value to questionsInput state
            onChange={(e) => setQuestionsInput(e.target.value)} // Updates state on textarea change
            placeholder={`What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?\nWhat is the waiting period for pre-existing diseases (PED) to be covered?\n...`} // Placeholder text
            aria-label="Questions"
            disabled={loading} // Disable textarea when loading
          ></textarea>
        </div>

        {/* Submit Button */}
        <button
          onClick={handleSubmit} // Calls handleSubmit function on click
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition duration-300 ease-in-out transform hover:scale-105"
          disabled={loading} // Disable button when loading
          aria-label="Submit Query"
        >
          {loading ? 'Processing...' : 'Retrieve Answers'} {/* Change button text based on loading state */}
        </button>

        {/* Loading Message */}
        {loading && (
          <div className="text-center text-blue-600 mt-4">
            <div className="flex items-center justify-center">
              {/* Spinner SVG for visual loading indicator */}
              <svg className="animate-spin h-5 w-5 mr-3 text-blue-600" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Retrieving answers from backend...
            </div>
          </div>
        )}
        {/* Error Message */}
        {error && (
          <div className="text-center text-red-600 mt-4 p-2 bg-red-100 rounded-lg border border-red-200">
            Error: {error}
          </div>
        )}

        {/* Response Display */}
        {response && (
          <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h2 className="text-xl font-semibold text-gray-800 mb-3">Retrieved Answers:</h2>
            {/* Display answers as an ordered list */}
            {response.answers && response.answers.length > 0 ? (
              <ol className="list-decimal list-inside space-y-2 text-sm text-gray-900">
                {response.answers.map((answer, index) => (
                  <li key={index} className="bg-gray-100 p-2 rounded-md">
                    <span className="font-medium text-gray-700">Q{index + 1}:</span> {questionsInput.split('\n').filter(q => q.trim() !== '')[index] || 'N/A'} <br/>
                    <span className="font-medium text-gray-700">A{index + 1}:</span> {answer}
                  </li>
                ))}
              </ol>
            ) : (
              <p className="text-gray-600">No answers retrieved.</p>
            )}

            {/* Copy to Clipboard Button */}
            {response.answers && response.answers.length > 0 && (
              <button
                onClick={copyToClipboard}
                className="mt-4 w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition duration-300 ease-in-out transform hover:scale-105"
                aria-label="Copy Answers to Clipboard"
              >
                Copy Answers
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
