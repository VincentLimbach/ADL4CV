import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exp_decay(x, a, b, c):
    return a * np.exp(- b * x + - c * x**2)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

loss_values = [
    0.03511909395456314, 0.027413280680775642, 0.025127600878477097, 0.023011228069663048,
    0.021396035328507423, 0.019672157242894173, 0.017899679020047188, 0.016470041126012802,
    0.01528116688132286, 0.014474868774414062, 0.013866662047803402, 0.013274410739541054,
    0.012840916402637959, 0.012476456351578236, 0.012113790027797222, 0.011826777830719948,
    0.011547677218914032, 0.011275921948254108, 0.010995343327522278, 0.010805811733007431,
    0.010593065991997719, 0.010361436754465103, 0.010204820893704891, 0.010046741925179958,
    0.00990748219192028, 0.00973215140402317, 0.009636863134801388, 0.009462562389671803,
    0.009328528307378292, 0.009205861948430538, 0.008492136374115944, 0.008364047855138779,
    0.008306880481541157, 0.008253112435340881, 0.008221069350838661
]

val_values = [
    0.029552210122346878, 0.02556503936648369, 0.023926004767417908, 0.021577948704361916,
    0.020019445568323135, 0.01823064684867859, 0.01673101633787155, 0.015559113584458828,
    0.014620767906308174, 0.013633190654218197, 0.013057533651590347, 0.012829052284359932,
    0.012454860843718052, 0.011950045824050903, 0.011782544665038586, 0.011308127082884312,
    0.011212402954697609, 0.010870065540075302, 0.010718943551182747, 0.010464970953762531,
    0.010365677997469902, 0.009898651391267776, 0.009862051345407963, 0.009846493601799011,
    0.009508136659860611, 0.009991045109927654, 0.009342341683804989, 0.009400959126651287,
    0.00899716466665268, 0.008935840800404549, 0.008266482502222061, 0.008059198968112469,
    0.008126288652420044, 0.008088851347565651, 0.007938170805573463
]

epochs = np.arange(1, len(loss_values) + 1)

popt_loss, _ = curve_fit(exp_decay, epochs, loss_values)
a_loss, b_loss, c_loss = popt_loss

popt_val, _ = curve_fit(exp_decay, epochs, val_values)
a_val, b_val, c_val = popt_val

print(f"Fitted parameters for loss: a = {a_loss:.4f}, b = {b_loss:.4f}")
print(f"Fitted parameters for validation loss: a = {a_val:.4f}, b = {b_val:.4f}")

future_epochs = np.arange(1, 51)  # Extrapolate to 50 epochs

extrapolated_loss = exp_decay(future_epochs, a_loss, b_loss, c_loss)
extrapolated_val = exp_decay(future_epochs, a_val, b_val, c_val)

plt.figure(figsize=(10, 5))

plt.plot(epochs, loss_values, 'o', label='Original Loss')
plt.plot(epochs, val_values, 'x', label='Original Validation Loss')

plt.plot(future_epochs, extrapolated_loss, '-', label='Extrapolated Loss')
plt.plot(future_epochs, extrapolated_val, '--', label='Extrapolated Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss and Validation Loss Extrapolation')
plt.legend()
plt.grid(True)
plt.show()
