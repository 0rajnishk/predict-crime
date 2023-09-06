import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { GoogleMap, LoadScript, Marker } from '@react-google-maps/api'; // Import components from the new library
import Cookies from 'js-cookie';

const Result = () => {
    const [mapLoaded, setMapLoaded] = useState(false);
    const location = useLocation();
    const searchParams = new URLSearchParams(location.search);

    const city = searchParams.get('city');
    const finalPredictedClass = searchParams.get('finalPredictedClass');
    const [mapCenter, setMapCenter] = useState({ lat: 34.0522, lng: -118.2437 }); // Default center for Los Angeles
    const [cityCoordinates, setCityCoordinates] = useState(null); // Store the city's coordinates
    // To get the historical crime data from the cookie
    const historicalCrimes = Cookies.get('historical_crimes');
    const historicalCrimesData = historicalCrimes ? JSON.parse(historicalCrimes) : null;



    const mapStyles = {
        height: '100%',
        width: '100%',
    };

    useEffect(() => {
        if (city) {
            fetch(`https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(city)},California,CA&key=AIzaSyC1RhaxHERmptd8axCftEWFi4t6NasiIcY`)
                .then(response => response.json())
                .then(data => {
                    if (data.results && data.results.length > 0) {
                        const { lat, lng } = data.results[0].geometry.location;
                        setCityCoordinates({ lat, lng }); // Store the city's coordinates
                        setMapCenter({ lat, lng }); // Set map center to city's coordinates
                        setMapLoaded(true); // Mark the map as loaded
                    }
                })
                .catch(error => {
                    console.error('Error fetching geolocation data:', error);
                });
        }
    }, [city]);


    const handleLoadMap = () => {
        // This function is called when the map is loaded. You can use it for any additional functionality.
        setMapLoaded(true);
    };

    const customMarkerIcon = () => {
        let iconUrl = 'https://cdn-icons-png.flaticon.com/512/3839/3839431.png'; // I am using this icon as default.

        switch (finalPredictedClass) {
            case 'ARSON':
                iconUrl = 'https://www.svgrepo.com/show/13210/flame.svg';
                break;
            case 'TRESPASS':
                iconUrl = 'https://static.thenounproject.com/png/4116425-200.png';
                break;
            case 'FRAUD - IMPERSONATION':
                iconUrl = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRL-rIHz7x5hAjHlrbxtyZrzsnp-9SjxMIy9Fp6oiUA3_3gTlJbTlvwQBWyu6-Ko9EYshU&usqp=CAU';
                break;
            case 'ROBBERY':
                iconUrl = 'https://static.thenounproject.com/png/641981-200.png';
                break;
            case 'BURGLARY - FORCED ENTRY':
                iconUrl = 'https://static.thenounproject.com/png/80200-200.png';
                break;
            case 'DAMAGE TO PROPERTY':
                iconUrl = 'https://www.svgrepo.com/show/314184/house-damage-solid.svg';
                break;
            case 'LARCENY - OTHER':
                iconUrl = 'https://cdn.iconscout.com/icon/premium/png-256-thumb/theft-2171452-1819965.png?f=webp';
                break;
            // Add more cases for other classes if needed
            default:
                break;
        }

        return {
            url: iconUrl,
            scaledSize: new window.google.maps.Size(80, 80), // Adjust the size as needed
        };
    };



    return (
        <div className='bg-gray-100 min-h-screen'>
            <div className='bg-[#0087B4] text-center py-8'>
                <h1 className='text-5xl font-semibold text-white'>Result Prediction</h1>
                <p className='text-white text-lg mt-3'>Based on historical data. Forecasted results might vary.</p>
                <a href="/"><h2>Click here to Predict more</h2></a>
            </div>
            <h1 style={{ color: 'red', fontSize: '30px', textAlign: 'center', margin: '0 auto' }}>{finalPredictedClass}</h1>

            <div className='mt-8 mx-auto w-[80%] h-[400px] rounded-lg shadow-md overflow-hidden'>
                <LoadScript
                    googleMapsApiKey="AIzaSyC1RhaxHERmptd8axCftEWFi4t6NasiIcY"
                    libraries={['places']}
                    onLoad={handleLoadMap}
                >
                    {mapLoaded && (
                        <GoogleMap
                            mapContainerStyle={mapStyles}
                            center={mapCenter}
                            zoom={14}
                        >
                            {/* Render your marker here */}
                            {cityCoordinates && (
                                <Marker
                                    position={cityCoordinates}
                                    icon={customMarkerIcon()}
                                />
                            )}
                        </GoogleMap>
                    )}
                </LoadScript>
            </div>


            <div className="mt-8 mx-auto w-[90%]">
                <h2 className="text-xl font-semibold mb-4">Historical Crime Data</h2>
                <table className="w-full border-collapse border border-gray-300">
                    <thead>
                        <tr className="bg-gray-200">
                            <th className="border border-gray-300 p-2">Address</th>
                            <th className="border border-gray-300 p-2">Incident Date</th>
                            <th className="border border-gray-300 p-2">Offense Category</th>
                            <th className="border border-gray-300 p-2">Neighborhood</th>
                        </tr>
                    </thead>
                    <tbody>
                        {historicalCrimes &&
                            historicalCrimesData.map((crime, index) => (
                                <tr key={index}>
                                    <td className="border border-gray-300 p-2">{crime.address}</td>
                                    <td className="border border-gray-300 p-2">{crime.incident_t}</td>
                                    <td className="border border-gray-300 p-2">{crime.offense_ca}</td>
                                    <td className="border border-gray-300 p-2">{crime.neighborhood}</td>
                                </tr>
                            ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default Result;
