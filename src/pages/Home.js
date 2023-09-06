import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import Cookies from 'js-cookie';


const Home = () => {
    const [selectedDate, setSelectedDate] = useState('');
    const [neighborhood, setNeighborhood] = useState('');
    const [predictionResult, setPredictionResult] = useState(null);

    const handlePrediction = async () => {
        const requestBody = {
            day: parseInt(selectedDate.split('-')[2]),
            month: parseInt(selectedDate.split('-')[1]),
            year: parseInt(selectedDate.split('-')[0]),
            neighborhood: neighborhood
        };

        try {
            const response = await fetch('https://predictcrime.azurewebsites.net/predict', {

                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            if (response.ok) {
                const data = await response.json();
                setPredictionResult(data.final_predicted_class);
                Cookies.set('historical_crimes', JSON.stringify(data.historical_crimes));


                // Redirect to Result page with data as URL parameters
                window.location.href = `/result?city=${neighborhood}&dnnPredictedClass=${data.dnn_predicted_class}&finalPredictedClass=${data.final_predicted_class}`;
            } else {
                console.error('Failed to get prediction');
            }
        } catch (error) {
            console.error('Error:', error);
            console.log(predictionResult)
        }
    };

    return (
        <div className='bg-[#0087B4] min-h-screen flex flex-col justify-center items-center'>
            <h1 className='text-6xl font-semibold text-white mb-6'>Machine Learning Prediction</h1>
            <p className='text-white md:text-lg mb-10'>Enter the date and neighborhood to get predictions.</p>

            <div className='bg-white rounded-lg p-4 border border-gray-300 w-96'>
                <div className='mb-3'>
                    <label className='block text-gray-600 text-sm mb-1'>Select Date</label>
                    <input
                        className='w-full border rounded-lg p-2'
                        type='date'
                        value={selectedDate}
                        onChange={(e) => setSelectedDate(e.target.value)}
                    />
                </div>
                <div className='mb-3'>
                    <label className='block text-gray-600 text-sm mb-1'>Input Location</label>
                    <input
                        className='w-full border rounded-lg p-2'
                        type='text'
                        placeholder='Enter location'
                        value={neighborhood}
                        onChange={(e) => setNeighborhood(e.target.value)}
                    />
                </div>
                <div className='text-center'>
                    <Link to='/result'>
                        <button
                            className='px-4 py-2 rounded-lg text-white bg-[#008CB8] hover:bg-[#0c9eca]'
                            onClick={handlePrediction}
                        >
                            Get Prediction
                        </button>
                    </Link>
                </div>
            </div>
        </div>
    );
};

export default Home;
