import joblib

model = joblib.load("ml/cnc_failure_model.pkl")

# Example CNC sensor values
temperature = 53.5
vibration = 0.174
spindle_rpm = 2528
motor_current = 3.56
tool_wear = 12.5

data = [[
    temperature,
    vibration,
    spindle_rpm,
    motor_current,
    tool_wear
]]

prediction = model.predict(data)[0]

if prediction == 0:
    status = "NORMAL"
else:
    status = "WARNING"

print("==============================")
print("CNC AI PREDICTION")
print("==============================")
print("Temperature  :", temperature)
print("Vibration    :", vibration)
print("Spindle RPM  :", spindle_rpm)
print("Motor Current:", motor_current)
print("Tool Wear    :", tool_wear)
print("------------------------------")
print("AI Prediction:", status)