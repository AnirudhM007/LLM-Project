import React, { useState } from 'react';

const App = () => {
  // State variables for document URL, questions input, and response
  const [documentUrl, setDocumentUrl] = useState('');
  const [questionsInput, setQuestionsInput] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Function to handle form submission when the "Retrieve Answers" button is clicked
  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResponse(null);

    // Split the questions input by new line to get an array of individual questions
    // Filter out any empty lines that might result from extra newlines
    const questions = questionsInput.split('\n').filter(q => q.trim() !== '');

    if (!documentUrl.trim() || questions.length === 0) {
        setError("Please provide a document URL and at least one question.");
        setLoading(false);
        return;
    }

    // Construct the payload to be sent to your backend API.
    const payload = {
      documents: documentUrl, // This sends the URL to your backend
      questions: questions,
    };

    try {
      // Define the URL for your FastAPI backend endpoint.
      const apiUrl = "http://localhost:8000/api/v1/hackrx/run";

      // Make the API call to your backend using fetch.
      const apiResponse = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Authorization': 'Bearer 5ba298f10582e591e01dd5a437f580a8da1354f64898b2042fc74b9e0968f9d1',
        },
        body: JSON.stringify(payload)
      });

      const result = await apiResponse.json();

      // Check if the API response was successful (status code 2xx)
      if (!apiResponse.ok) {
        throw new Error(result.detail || 'Failed to retrieve answers from backend.');
      }
      
      // Combine questions with answers for display
      const formattedResponse = {
          ...result,
          questions: questions
      };
      setResponse(formattedResponse);

    } catch (err) {
      // Catch any errors during the fetch operation or response processing
      setError('Failed to fetch response from backend: ' + err.message);
      console.error("Error during API call:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4 font-sans">
      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-3xl">
        <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          VALORetrieve
        </h1>

        {/* Document URL Input field */}
        <div className="mb-4">
          <label htmlFor="documentUrl" className="block text-gray-700 text-sm font-medium mb-2">
            Document URL
          </label>
          <input
            type="text"
            id="documentUrl"
            className="shadow-sm appearance-none border rounded-lg w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={documentUrl}
            onChange={(e) => setDocumentUrl(e.target.value)}
            placeholder="e.g., https://example.com/policy.pdf"
            aria-label="Document URL"
            disabled={loading}
          />
        </div>

        {/* Questions Input textarea */}
        <div className="mb-6">
          <label htmlFor="questions" className="block text-gray-700 text-sm font-medium mb-2">
            Questions (one per line)
          </label>
          <textarea
            id="questions"
            className="shadow-sm appearance-none border rounded-lg w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-blue-500 h-48"
            value={questionsInput}
            onChange={(e) => setQuestionsInput(e.target.value)}
            placeholder="What is the grace period...?\nWhat is the waiting period...?"
            aria-label="Questions"
            disabled={loading}
          ></textarea>
        </div>

        {/* Submit Button */}
        <button
          onClick={handleSubmit}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition duration-300"
          disabled={loading}
          aria-label="Submit Query"
        >
          {loading ? 'Processing...' : 'Retrieve Answers'}
        </button>

        {/* Loading Message */}
        {loading && (
          <div className="text-center text-blue-600 mt-4">
            <div className="flex items-center justify-center">
              <svg className="animate-spin h-5 w-5 mr-3 text-blue-600" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing document and retrieving answers... This may take up to 30 seconds.
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
            {response.answers && response.answers.length > 0 ? (
              <div className="space-y-4 text-sm text-gray-900">
                {response.answers.map((answer, index) => (
                  <div key={index} className="bg-gray-100 p-3 rounded-md">
                    <p className="font-medium text-gray-700">Q: {response.questions[index]}</p>
                    <p className="mt-1 whitespace-pre-wrap">A: {answer}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-600">No answers retrieved.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
