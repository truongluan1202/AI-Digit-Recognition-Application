const ACTIVATION_RATE = 0.07;

let numbers = [],
  topology = [],
  weight = [],
  outputVals = [];

function readTextFile(file) {
  var rawFile = new XMLHttpRequest();
  rawFile.open("GET", file, false);
  rawFile.onreadystatechange = function () {
    if (rawFile.readyState === 4) {
      if (rawFile.status === 200 || rawFile.status == 0) {
        var allText = rawFile.responseText;
        numbers = allText.split(/\s+/).map(Number);
      }
    }
  };
  rawFile.send(null);
}

// INPUT NET ****************************** //
readTextFile("net.dat");
var n = numbers[0];
for (let i = 0; i < n; ++i) {
  topology.push(numbers[i + 1]);
}
let cnt = n + 1;
for (let i = 0; i < n - 1; ++i) {
  let tmpi = [];
  for (let j = 0; j < topology[i] + 1; ++j) {
    let tmpj = [];
    for (let k = 0; k < topology[i + 1]; ++k) {
      tmpj.push(numbers[cnt]);
      cnt++;
    }
    tmpi.push(tmpj);
  }
  weight.push(tmpi);
}

// MOUSE MOVEMENT ************************* //

const canvas = document.querySelector("#canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.querySelector("#clear-bg");
const predictedNumber = document.querySelector("#predicted-number");
const scaleCanvas = document.querySelector("#scaling-canvas");
const scaleCtx = scaleCanvas.getContext("2d");

let isDrawing = false,
  isErase = false,
  drawColor = "#fff",
  eraseColor = "#000",
  lineWidth = 10;

const setCanvasBackground = () => {
  ctx.fillStyle = eraseColor;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  scaleCtx.fillStyle = eraseColor;
  scaleCtx.fillRect(0, 0, scaleCanvas.width, scaleCanvas.height);
};

function reSize() {
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  lineWidth = (canvas.width * 17) / 500;
  scaleCanvas.width = scaleCanvas.height = 28;
}

// window.addEventListener("resize", reSize);

window.addEventListener("load", () => {
  // setting canvas w/h == offsetw/h
  reSize();
  setCanvasBackground();
});

window.onresize = function () {
  reSize();
};

canvas.addEventListener("mousedown", (e) => {
  if (e.which === 1) {
    isDrawing = true;
  }
  if (e.which === 3) {
    isErase = true;
  }
  ctx.beginPath();
});

const drawRect = (e) => {
  ctx.fillRect(
    e.offsetX - lineWidth / 2,
    e.offsetY - lineWidth / 2,
    lineWidth,
    lineWidth
  );
};

canvas.addEventListener("mousemove", (e) => {
  if (isDrawing) {
    ctx.fillStyle = drawColor;
    drawRect(e);
  }
  if (isErase) {
    ctx.fillStyle = eraseColor;
    drawRect(e);
  }
});

canvas.addEventListener("mouseup", (e) => {
  isDrawing = false;
  isErase = false;
  const imgData = getImageData();
  predictedNumber.innerText = getResult(imgData);
});

clearBtn.addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  setCanvasBackground();
  predictedNumber.innerText = "...";
});

window.addEventListener("keydown", (e) => {
  if (e.key === "c" || e.key === "C") {
    clearBtn.click();
  }
});

function getImageData() {
  let imgData = scaleImageData();
  let pixels = imgData.data;
  var result1 = [];
  var result = [];

  for (let i = 0; i < pixels.length; i += 4) {
    const red = pixels[i];
    const green = pixels[i + 1];
    const blue = pixels[i + 2];
    const num = (red + green + blue) / 3;
    result1.push(num);
    result.push(0);
  }

  var dx = 27,
    dy = 27;
  for (let x = 0; x < 28; ++x)
    for (let y = 0; y < 28; ++y)
      if (result1[x * 28 + y] > 0) {
        dx = Math.min(dx, x);
        dy = Math.min(dy, y);
      }

  for (let x = dx; x < 28; ++x)
    for (let y = dy; y < 28; ++y) {
      var newPos = (x - dx) * 28 + (y - dy);
      var oldPos = x * 28 + y;
      result[newPos] = result1[oldPos];
    }
  return result;
}

function scaleImageData() {
  // Canvas for scaling
  scaleCtx.drawImage(canvas, 0, 0, 28, 28);
  var scaledImageData = scaleCtx.getImageData(
    0,
    0,
    scaleCanvas.width,
    scaleCanvas.height
  );
  // scaleCtx.scale(1 / scale, 1 / scale);
  return scaledImageData;
}

function activationFunction(x) {
  return Math.max(x, 0.0) * ACTIVATION_RATE;
}

function getResult(inputVals) {
  let tmp = [];
  outputVals = [];
  for (let i = 0; i < topology[0]; ++i) {
    tmp.push(inputVals[i] / 255);
  }
  tmp.push(1); // bias neuron
  outputVals.push(tmp);

  for (let i = 1; i < n; ++i) {
    let tmpi = [];
    for (let j = 0; j < topology[i]; ++j) {
      let sumWeightedInput = 0.0;

      for (let k = 0; k < topology[i - 1] + 1; ++k) {
        let w = weight[i - 1][k][j];
        sumWeightedInput += outputVals[i - 1][k] * w;
      }

      tmpi.push(activationFunction(sumWeightedInput));
    }
    tmpi.push(1);
    outputVals.push(tmpi);
  }

  let maxVal = -10000000;
  let result = -1;
  for (let i = 0; i < topology[n - 1]; ++i) {
    if (outputVals[n - 1][i] > maxVal) {
      maxVal = outputVals[n - 1][i];
      result = i;
    }
  }

  if (maxVal == 0) return "...";
  return result;
}
