import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom'
import PredictPage from './pages/PredictPage'
import DashboardPage from './pages/DashboardPage'
import PerformancePage from './pages/PerformancePage'
import HistoryPage from './pages/HistoryPage'

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        {/* Navigation */}
        <nav className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4">
            <div className="flex items-center justify-between h-16">
              <h1 className="text-xl font-bold text-gray-900">
                Kickstarter Predictor
              </h1>
              <div className="flex space-x-4">
                <NavLink
                  to="/"
                  className={({ isActive }) =>
                    `px-3 py-2 rounded-md text-sm font-medium ${
                      isActive
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-600 hover:text-gray-900'
                    }`
                  }
                >
                  Predict
                </NavLink>
                <NavLink
                  to="/dashboard"
                  className={({ isActive }) =>
                    `px-3 py-2 rounded-md text-sm font-medium ${
                      isActive
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-600 hover:text-gray-900'
                    }`
                  }
                >
                  Dashboard
                </NavLink>
                <NavLink
                  to="/performance"
                  className={({ isActive }) =>
                    `px-3 py-2 rounded-md text-sm font-medium ${
                      isActive
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-600 hover:text-gray-900'
                    }`
                  }
                >
                  Performance
                </NavLink>
                <NavLink
                  to="/history"
                  className={({ isActive }) =>
                    `px-3 py-2 rounded-md text-sm font-medium ${
                      isActive
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-600 hover:text-gray-900'
                    }`
                  }
                >
                  History
                </NavLink>
              </div>
            </div>
          </div>
        </nav>

        {/* Page Content */}
        <main className="max-w-7xl mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<PredictPage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/performance" element={<PerformancePage />} />
            <Route path="/history" element={<HistoryPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
