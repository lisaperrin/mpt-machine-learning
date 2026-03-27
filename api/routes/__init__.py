from api.routes.optimization import router as optimization_router
from api.routes.backtest import router as backtest_router
from api.routes.analysis import router as analysis_router
from api.routes.simulation import router as simulation_router
from api.routes.data import router as data_router
from api.routes.evaluation import router as evaluation_router
from api.routes.health import router as health_router

all_routers = [
    optimization_router,
    backtest_router,
    analysis_router,
    simulation_router,
    data_router,
    evaluation_router,
    health_router,
]
