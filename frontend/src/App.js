import './App.css';
import { RouterProvider, createBrowserRouter } from 'react-router-dom';
import Home from './pages/Home';
import Result from './pages/Result.js';


function App() {
  const router = createBrowserRouter([
    {
      path: '/',
      element: <Home />
    },
    {
      path: '/result',
      element: <Result />
    }
  ])
  return (
    <RouterProvider router={router} />
  );
}

export default App;
