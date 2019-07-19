const NUM_BIRDS = 500;
let showingBirds = [];
let score = 0;
let pipes = [];
let clicked = 0;
let paused = false;
let slider;
let generation = 1;
let prevBest = null;
let showingPrev = false;
let button;
let button2;

function setup() {
    createCanvas(284 * 2, 512);
    slider = createSlider(1, 200, 1);
    button = createButton('Show previous best');
    button2 = createButton('Kill all and restart');
    pipes.push(new Pipes());
    for (i = 0; i < NUM_BIRDS; i++) {
        showingBirds.push(new Bird());
    }
}

function toggleShowPrev() {
    if (showingPrev != null) {
        showingPrev = !showingPrev;
    }
}

function draw() {
    for (let z = 0; z < slider.value(); z++) {
        let index;
        let xDistRecord = Infinity;
        for (let i = 0; i < pipes.length; i++) { // finds index of closest pipe
            xDist = pipes[i].x - 100 + 90;
            if (xDist > 0 && xDist < xDistRecord) {
                xDistRecord = xDist;
                index = i;
            }
        }
        fill(color(135, 206, 235));
        noStroke();
        rect(0, 0, width, height); // background
        pipes.forEach(p => {
            showingBirds.forEach(b => {
                if (p.colision(b) || b.y < 0 || b.y > 430) { // checks collisions for all pipes
                    //console.log(b.fit);
                    if (showingBirds.length == 1) {
                        restart(b);
                    }
                    showingBirds.splice(showingBirds.indexOf(b), 1)
                }
            });
            p.show();
            p.update();

        })
        if (pipes[0].x < -60) { // removes pipe when off screen
            pipes.shift();
        }
        if (pipes[0].x == 90) { // inciments score when pipe is behind bird
            score++;
        }
        fill(color(247, 233, 181));
        rect(0, height - 70, width, 70); // ground
        button.mousePressed(toggleShowPrev);
        button2.mousePressed(restart);
        showingBirds.forEach(b => {
            b.shouldFlap(pipes[index])
            b.update();
            if (showingPrev) {
                showingBirds[0].show();
            } else {
                b.show();
            }
        });

        fill(255);
        textSize(30);
        text(score, width / 2, 50);
        fill(0);
        textSize(20);
        text(`${showingBirds.length} / ${NUM_BIRDS} Birds left`, 5, height - 40);
        text(`Generation ${generation}`, 5, height - 10);

    }
}

function newPipe() {
    pipes.push(new Pipes());
}

function restart(bird) {
    if (bird == undefined) {
        bird = showingBirds[0];
    }
    generation++;
    console.log('Highest fitness: ' + bird.fit + '\n' + 'begining gen ' + generation + '\n');
    score = 0;
    pipes = [new Pipes()];
    showingBirds = [];
    if (prevBest != null) {
        showingBirds.push(prevBest);
    }
    prevBest = bird;
    for (i = 0; i < NUM_BIRDS - 1; i++) {
        showingBirds.push(new Bird(bird));
    }
}



function mutate(x) {
    if (random(1) < 0.1) {
        let offset = randomGaussian() * 0.5;
        let newx = x + offset;
        return newx;
    } else {
        return x;
    }
}

function sig(x) { // sigmoid used for normalizing Bird.yVel
    let num = 1 / (1 + Math.exp(x / 3));
    return num;
}

class Bird {

    constructor(b) {
        if (b) {
            this.net = b.net.copy();
            this.net.mutate(mutate);
        } else {
            this.net = new NeuralNetwork(4, 6, 1);
        }
    }

    fit = 0;
    x = 100;
    y = 200;
    grav = .4;
    yVel = 0;
    width = 30;


    show() {
        ellipseMode(CENTER);
        fill(color(255, 255, 0, 80));
        stroke(0);
        ellipse(this.x, this.y, this.width, this.width);
        noStroke();
    }

    flap() {
        if (this.yVel > 0) {
            this.yVel = -8;
        }
    }

    mutate(b) {

    }

    shouldFlap(p) {
        if (this.yVel > 0 && this.net.predict(p.getDist(this)) >= .5) {
            this.flap();
        }
    }

    update() {
        this.fit++;
        this.yVel += this.grav;
        this.y += this.yVel;
        if (this.y > 450) {
            this.y = 300;
        }
    }
}


class Pipes {
    pipeGap = 150;
    speed = 2;
    delay = 350; // must be evenly divisible by speed

    pWidth = 60;
    pHeight = 350;

    maxY = 400;
    minY = 160;

    x = width + 60;
    y = Math.random() * (this.maxY - this.minY) + this.minY;

    pColor = color(27, 227, 61);

    setColor(r, g, b) {
        this.pColor = color(r, g, b);
    }

    show() {
        fill(this.pColor);
        rect(this.x, this.y - this.pHeight - this.pipeGap, this.pWidth, this.pHeight); // top pipe 
        rect(this.x, this.y, this.pWidth, this.pHeight); // bottom pipe

    }

    colision({ x, y, width }) { // detects if bird and either pipe are intersecting
        return collideRectCircle(this.x, this.y - this.pipeGap - this.pHeight, this.pWidth, this.pHeight, x, y, width) || collideRectCircle(this.x, this.y, this.pWidth, this.pHeight, x, y, width); // bottom pipe

    }

    lastYVel = 0;
    getDist(bird) {
        let xDist = this.x - bird.x; // bird x and y are reletive to center of circle
        let bDist = this.y - bird.y; // y dist to top of bottom pipe
        let tDist = this.y - bird.y - this.pipeGap; // y dist to bottom of top pipe
        let yVel = this.lastYVel;

        this.lastYVel = sig(bird.yVel);

        return [xDist / width, bDist / 412, tDist / 412, yVel];
    }

    update() {
        this.x -= this.speed;
        if (this.x == this.delay) {
            newPipe();
            //console.log('New pipe');
        }
    }
}

class ActivationFunction {
    constructor(func, dfunc) {
        this.func = func;
        this.dfunc = dfunc;
    }
}

let sigmoid = new ActivationFunction(
    x => 1 / (1 + Math.exp(-x)),
    y => y * (1 - y)
);

let tanh = new ActivationFunction(
    x => Math.tanh(x),
    y => 1 - (y * y)
);


class NeuralNetwork {
    /*
    * if first argument is a NeuralNetwork the constructor clones it
    * USAGE: cloned_nn = new NeuralNetwork(to_clone_nn);
    */
    constructor(in_nodes, hid_nodes, out_nodes) {
        if (in_nodes instanceof NeuralNetwork) {
            let a = in_nodes;
            this.input_nodes = a.input_nodes;
            this.hidden_nodes = a.hidden_nodes;
            this.output_nodes = a.output_nodes;

            this.weights_ih = a.weights_ih.copy();
            this.weights_ho = a.weights_ho.copy();

            this.bias_h = a.bias_h.copy();
            this.bias_o = a.bias_o.copy();
        } else {
            this.input_nodes = in_nodes;
            this.hidden_nodes = hid_nodes;
            this.output_nodes = out_nodes;

            this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
            this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
            this.weights_ih.randomize();
            this.weights_ho.randomize();

            this.bias_h = new Matrix(this.hidden_nodes, 1);
            this.bias_o = new Matrix(this.output_nodes, 1);
            this.bias_h.randomize();
            this.bias_o.randomize();
        }

        // TODO: copy these as well
        this.setLearningRate();
        this.setActivationFunction();


    }

    predict(input_array) {
        // Generating the Hidden Outputs
        let inputs = Matrix.fromArray(input_array);
        let hidden = Matrix.multiply(this.weights_ih, inputs);
        hidden.add(this.bias_h);
        // activation function!
        hidden.map(this.activation_function.func);

        // Generating the output's output!
        let output = Matrix.multiply(this.weights_ho, hidden);
        output.add(this.bias_o);
        output.map(this.activation_function.func);

        // Sending back to the caller!
        return output.toArray();
    }

    setLearningRate(learning_rate = 0.1) {
        this.learning_rate = learning_rate;
    }

    setActivationFunction(func = sigmoid) {
        this.activation_function = func;
    }

    train(input_array, target_array) {
        // Generating the Hidden Outputs
        let inputs = Matrix.fromArray(input_array);
        let hidden = Matrix.multiply(this.weights_ih, inputs);
        hidden.add(this.bias_h);
        // activation function!
        hidden.map(this.activation_function.func);

        // Generating the output's output!
        let outputs = Matrix.multiply(this.weights_ho, hidden);
        outputs.add(this.bias_o);
        outputs.map(this.activation_function.func);

        // Convert array to matrix object
        let targets = Matrix.fromArray(target_array);

        // Calculate the error
        // ERROR = TARGETS - OUTPUTS
        let output_errors = Matrix.subtract(targets, outputs);

        // let gradient = outputs * (1 - outputs);
        // Calculate gradient
        let gradients = Matrix.map(outputs, this.activation_function.dfunc);
        gradients.multiply(output_errors);
        gradients.multiply(this.learning_rate);


        // Calculate deltas
        let hidden_T = Matrix.transpose(hidden);
        let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);

        // Adjust the weights by deltas
        this.weights_ho.add(weight_ho_deltas);
        // Adjust the bias by its deltas (which is just the gradients)
        this.bias_o.add(gradients);

        // Calculate the hidden layer errors
        let who_t = Matrix.transpose(this.weights_ho);
        let hidden_errors = Matrix.multiply(who_t, output_errors);

        // Calculate hidden gradient
        let hidden_gradient = Matrix.map(hidden, this.activation_function.dfunc);
        hidden_gradient.multiply(hidden_errors);
        hidden_gradient.multiply(this.learning_rate);

        // Calcuate input->hidden deltas
        let inputs_T = Matrix.transpose(inputs);
        let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

        this.weights_ih.add(weight_ih_deltas);
        // Adjust the bias by its deltas (which is just the gradients)
        this.bias_h.add(hidden_gradient);

        // outputs.print();
        // targets.print();
        // error.print();
    }

    serialize() {
        return JSON.stringify(this);
    }

    static deserialize(data) {
        if (typeof data == 'string') {
            data = JSON.parse(data);
        }
        let nn = new NeuralNetwork(data.input_nodes, data.hidden_nodes, data.output_nodes);
        nn.weights_ih = Matrix.deserialize(data.weights_ih);
        nn.weights_ho = Matrix.deserialize(data.weights_ho);
        nn.bias_h = Matrix.deserialize(data.bias_h);
        nn.bias_o = Matrix.deserialize(data.bias_o);
        nn.learning_rate = data.learning_rate;
        return nn;
    }


    // Adding function for neuro-evolution
    copy() {
        return new NeuralNetwork(this);
    }

    // Accept an arbitrary function for mutation
    mutate(func) {
        this.weights_ih.map(func);
        this.weights_ho.map(func);
        this.bias_h.map(func);
        this.bias_o.map(func);
    }
}

class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array(this.rows).fill().map(() => Array(this.cols).fill(0));
    }

    copy() {
        let m = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                m.data[i][j] = this.data[i][j];
            }
        }
        return m;
    }

    static fromArray(arr) {
        return new Matrix(arr.length, 1).map((e, i) => arr[i]);
    }

    static subtract(a, b) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            console.log('Columns and Rows of A must match Columns and Rows of B.');
            return;
        }

        // Return a new Matrix a-b
        return new Matrix(a.rows, a.cols)
            .map((_, i, j) => a.data[i][j] - b.data[i][j]);
    }

    toArray() {
        let arr = [];
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                arr.push(this.data[i][j]);
            }
        }
        return arr;
    }

    randomize() {
        return this.map(e => Math.random() * 2 - 1);
    }

    add(n) {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) {
                console.log('Columns and Rows of A must match Columns and Rows of B.');
                return;
            }
            return this.map((e, i, j) => e + n.data[i][j]);
        } else {
            return this.map(e => e + n);
        }
    }

    static transpose(matrix) {
        return new Matrix(matrix.cols, matrix.rows)
            .map((_, i, j) => matrix.data[j][i]);
    }

    static multiply(a, b) {
        // Matrix product
        if (a.cols !== b.rows) {
            console.log('Columns of A must match rows of B.');
            return;
        }

        return new Matrix(a.rows, b.cols)
            .map((e, i, j) => {
                // Dot product of values in col
                let sum = 0;
                for (let k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                return sum;
            });
    }

    multiply(n) {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) {
                console.log('Columns and Rows of A must match Columns and Rows of B.');
                return;
            }

            // hadamard product
            return this.map((e, i, j) => e * n.data[i][j]);
        } else {
            // Scalar product
            return this.map(e => e * n);
        }
    }

    map(func) {
        // Apply a function to every element of matrix
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                let val = this.data[i][j];
                this.data[i][j] = func(val, i, j);
            }
        }
        return this;
    }

    static map(matrix, func) {
        // Apply a function to every element of matrix
        return new Matrix(matrix.rows, matrix.cols)
            .map((e, i, j) => func(matrix.data[i][j], i, j));
    }

    print() {
        console.table(this.data);
        return this;
    }

    serialize() {
        return JSON.stringify(this);
    }

    static deserialize(data) {
        if (typeof data == 'string') {
            data = JSON.parse(data);
        }
        let matrix = new Matrix(data.rows, data.cols);
        matrix.data = data.data;
        return matrix;
    }
}

if (typeof module !== 'undefined') {
    module.exports = Matrix;
}

/*
Repo: https://github.com/bmoren/p5.collide2D/
Created by http://benmoren.com
Some functions and code modified version from http://www.jeffreythompson.org/collision-detection
Version 0.6 | Nov 28th, 2018
CC BY-NC-SA 4.0
*/

p5.prototype._collideDebug = !1, p5.prototype.collideDebug = function (i) { _collideDebug = i }, p5.prototype.collideRectRect = function (i, t, e, o, r, l, n, c) { return i + e >= r && i <= r + n && t + o >= l && t <= l + c }, p5.prototype.collideRectCircle = function (i, t, e, o, r, l, n) { var c = r, p = l; return r < i ? c = i : r > i + e && (c = i + e), l < t ? p = t : l > t + o && (p = t + o), this.dist(r, l, c, p) <= n / 2 }, p5.prototype.collideCircleCircle = function (i, t, e, o, r, l) { return this.dist(i, t, o, r) <= e / 2 + l / 2 }, p5.prototype.collidePointCircle = function (i, t, e, o, r) { return this.dist(i, t, e, o) <= r / 2 }, p5.prototype.collidePointEllipse = function (i, t, e, o, r, l) { var n = r / 2, c = l / 2; if (i > e + n || i < e - n || t > o + c || t < o - c) return !1; var p = i - e, s = t - o, d = c * this.sqrt(this.abs(n * n - p * p)) / n; return s <= d && s >= -d }, p5.prototype.collidePointRect = function (i, t, e, o, r, l) { return i >= e && i <= e + r && t >= o && t <= o + l }, p5.prototype.collidePointLine = function (i, t, e, o, r, l, n) { var c = this.dist(i, t, e, o), p = this.dist(i, t, r, l), s = this.dist(e, o, r, l); return void 0 === n && (n = .1), c + p >= s - n && c + p <= s + n }, p5.prototype.collideLineCircle = function (i, t, e, o, r, l, n) { var c = this.collidePointCircle(i, t, r, l, n), p = this.collidePointCircle(e, o, r, l, n); if (c || p) return !0; var s = i - e, d = t - o, u = this.sqrt(s * s + d * d), h = ((r - i) * (e - i) + (l - t) * (o - t)) / this.pow(u, 2), y = i + h * (e - i), f = t + h * (o - t); return !!this.collidePointLine(y, f, i, t, e, o) && (this._collideDebug && this.ellipse(y, f, 10, 10), s = y - r, d = f - l, this.sqrt(s * s + d * d) <= n / 2) }, p5.prototype.collideLineLine = function (i, t, e, o, r, l, n, c, p) { var s = ((n - r) * (t - l) - (c - l) * (i - r)) / ((c - l) * (e - i) - (n - r) * (o - t)), d = ((e - i) * (t - l) - (o - t) * (i - r)) / ((c - l) * (e - i) - (n - r) * (o - t)); if (s >= 0 && s <= 1 && d >= 0 && d <= 1) { if (this._collideDebug || p) var u = i + s * (e - i), h = t + s * (o - t); return this._collideDebug && this.ellipse(u, h, 10, 10), !p || { x: u, y: h } } return !!p && { x: !1, y: !1 } }, p5.prototype.collideLineRect = function (i, t, e, o, r, l, n, c, p) { var s, d, u, h, y; return p ? (s = this.collideLineLine(i, t, e, o, r, l, r, l + c, !0), d = this.collideLineLine(i, t, e, o, r + n, l, r + n, l + c, !0), u = this.collideLineLine(i, t, e, o, r, l, r + n, l, !0), h = this.collideLineLine(i, t, e, o, r, l + c, r + n, l + c, !0), y = { left: s, right: d, top: u, bottom: h }) : (s = this.collideLineLine(i, t, e, o, r, l, r, l + c), d = this.collideLineLine(i, t, e, o, r + n, l, r + n, l + c), u = this.collideLineLine(i, t, e, o, r, l, r + n, l), h = this.collideLineLine(i, t, e, o, r, l + c, r + n, l + c)), !!(s || d || u || h) && (!p || y) }, p5.prototype.collidePointPoly = function (i, t, e) { for (var o = !1, r = 0, l = 0; l < e.length; l++) { r = l + 1, r == e.length && (r = 0); var n = e[l], c = e[r]; (n.y > t && c.y < t || n.y < t && c.y > t) && i < (c.x - n.x) * (t - n.y) / (c.y - n.y) + n.x && (o = !o) } return o }, p5.prototype.collideCirclePoly = function (i, t, e, o, r) { void 0 == r && (r = !1); for (var l = 0, n = 0; n < o.length; n++) { l = n + 1, l == o.length && (l = 0); var c = o[n], p = o[l]; if (this.collideLineCircle(c.x, c.y, p.x, p.y, i, t, e)) return !0 } if (1 == r) { if (this.collidePointPoly(i, t, o)) return !0 } return !1 }, p5.prototype.collideRectPoly = function (i, t, e, o, r, l) { void 0 == l && (l = !1); for (var n = 0, c = 0; c < r.length; c++) { n = c + 1, n == r.length && (n = 0); var p = r[c], s = r[n]; if (this.collideLineRect(p.x, p.y, s.x, s.y, i, t, e, o)) return !0; if (1 == l) { if (this.collidePointPoly(i, t, r)) return !0 } } return !1 }, p5.prototype.collideLinePoly = function (i, t, e, o, r) { for (var l = 0, n = 0; n < r.length; n++) { l = n + 1, l == r.length && (l = 0); var c = r[n].x, p = r[n].y, s = r[l].x, d = r[l].y; if (this.collideLineLine(i, t, e, o, c, p, s, d)) return !0 } return !1 }, p5.prototype.collidePolyPoly = function (i, t, e) { void 0 == e && (e = !1); for (var o = 0, r = 0; r < i.length; r++) { o = r + 1, o == i.length && (o = 0); var l = i[r], n = i[o], c = this.collideLinePoly(l.x, l.y, n.x, n.y, t); if (c) return !0; if (1 == e && (c = this.collidePointPoly(t[0].x, t[0].y, i))) return !0 } return !1 }, p5.prototype.collidePointTriangle = function (i, t, e, o, r, l, n, c) { var p = this.abs((r - e) * (c - o) - (n - e) * (l - o)); return this.abs((e - i) * (l - t) - (r - i) * (o - t)) + this.abs((r - i) * (c - t) - (n - i) * (l - t)) + this.abs((n - i) * (o - t) - (e - i) * (c - t)) == p }, p5.prototype.collidePointPoint = function (i, t, e, o, r) { return void 0 == r && (r = 0), this.dist(i, t, e, o) <= r }, p5.prototype.collidePointArc = function (i, t, e, o, r, l, n, c) { void 0 == c && (c = 0); var p = this.createVector(i, t), s = this.createVector(e, o), d = this.createVector(r, 0).rotate(l), u = p.copy().sub(s); if (p.dist(s) <= r + c) { var h = d.dot(u), y = d.angleBetween(u); if (h > 0 && y <= n / 2 && y >= -n / 2) return !0 } return !1 };
