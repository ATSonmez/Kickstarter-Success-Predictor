from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.sql import func
from database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    country = Column(String, nullable=False)
    goal = Column(Float, nullable=False)
    name_len = Column(Integer, nullable=False)
    blurb_len = Column(Integer, nullable=False)
    duration_days = Column(Integer, nullable=False)
    prep_days = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    prediction = Column(Boolean, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    auc_score = Column(Float, nullable=False)
    trained_at = Column(DateTime, server_default=func.now())
