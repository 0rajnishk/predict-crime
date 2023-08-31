import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { GoogleMap, LoadScript } from '@react-google-maps/api';

const Result = () => {
    const [mapLoaded, setMapLoaded] = useState(false);
    const location = useLocation();
    const searchParams = new URLSearchParams(location.search);

    const city = searchParams.get('city');
    const finalPredictedClass = searchParams.get('finalPredictedClass');
    const [mapCenter, setMapCenter] = useState({ lat: 34.0522, lng: -118.2437 }); // Default center for Los Angeles

    const mapStyles = {
        height: '100%',
        width: '100%',
    };

    // Use the Google Geocoding API to get the coordinates for the selected city (neighborhood)
    const [zoomLevel, setZoomLevel] = useState(14);
    useEffect(() => {
        if (city) {
            fetch(`https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(city)},Los+Angeles,CA&key=AIzaSyC1RhaxHERmptd8axCftEWFi4t6NasiIcY`)
                .then(response => response.json())
                .then(data => {
                    if (data.results && data.results.length > 0) {
                        const { lat, lng } = data.results[0].geometry.location;
                        setMapCenter({ lat, lng });
                        setZoomLevel(18); // Adjust the zoom level for a closer view
                    }
                })
                .catch(error => {
                    console.error('Error fetching geolocation data:', error);
                });
        }
    }, [city]);

    const handleLoadMap = () => {
        setMapLoaded(true);
    };

    const CustomMarker = () => {
        let iconUrl = 'https://cdn-icons-png.flaticon.com/512/3839/3839431.png';

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

        return (
            <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
                <img
                    src={iconUrl}
                    alt={finalPredictedClass}
                    style={{ width: '100px', height: '100px', marginBottom: '10px' }}
                />
                <div style={{ color: 'red', fontWeight: 'bold' }}>
                    {finalPredictedClass}
                </div>
            </div>
        );
    };




    return (
        <div className='bg-gray-100 min-h-screen'>

            <div className='bg-[#0087B4] text-center py-16'>
                <h1 className='text-5xl font-semibold text-white'>Result Prediction</h1>
                <p className='text-white text-lg mt-3'>Based on historical data. Forecasted results might vary.</p>             </div>
            <div className='mt-8 mx-auto w-[90%] h-[400px] rounded-lg shadow-md overflow-hidden'>
                <LoadScript
                    googleMapsApiKey="AIzaSyC1RhaxHERmptd8axCftEWFi4t6NasiIcY"
                    libraries={['places']}
                    onLoad={handleLoadMap}
                >
                    {mapLoaded && (
                        <GoogleMap
                            mapContainerStyle={mapStyles}
                            center={mapCenter}
                            zoom={zoomLevel} // Use the updated zoom level
                        >
                            {/* Render your marker here */}
                            <CustomMarker lat={mapCenter.lat} lng={mapCenter.lng} />
                        </GoogleMap>
                    )}
                </LoadScript>
            </div>


        </div>
    );
}

export default Result;
