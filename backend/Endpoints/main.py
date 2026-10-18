from fastapi import FastAPI, Depends, APIRouter
from fastapi.responses import JSONResponse
from psycopg_pool import PoolTimeout, TooManyRequests
from psycopg import OperationalError
import logging
from contextlib import asynccontextmanager
from db import pool, get_cursor
from auth import get_current_user, require_admin
from routers import auth_routes, query_routes, user_routes, admin_routes



@asynccontextmanager
async def lifespan(app):
    yield
    pool.close()


app = FastAPI(lifespan=lifespan) # lifespan closes the db pool on shutdown



@app.exception_handler(TooManyRequests)
@app.exception_handler(PoolTimeout)
def handle_pool_busy(request,exc):
    return JSONResponse(status_code=503, content={"detail": "Server busy, try again later"}, headers={"Retry-After": "5"})

@app.exception_handler(OperationalError)
def handle_db_unavailable(request,exc):
    logging.error("Database error", exc_info=exc)
    return JSONResponse(status_code=503, content={"detail": "Database unavailable"})

#app and db check
@app.get("/health/db")
def health_db():
    with get_cursor() as cur:
        cur.execute("SELECT 1")
    return {"status": "ok"}


api = APIRouter(prefix="/api")
api.include_router(auth_routes.router)
api.include_router(query_routes.router, dependencies=[Depends(get_current_user)])
api.include_router(user_routes.router, dependencies=[Depends(get_current_user)])
api.include_router(admin_routes.router, dependencies=[Depends(require_admin)])
app.include_router(api)





