var globalclicktime = 0;
let arrowId = 0;
let componentWidth = 120;
let componentHeight = 180;
var tool = 0;
// let isDarggingCanvas = false;
// let isDarggingElement = false;
// let canvasStartX,canvasStartY,canvasOffSetX,canvasOffSetY

let canvas = document.getElementById("canvas");
function sign(x) {
    return (x > 0) - (x < 0) || +x;
}
canvas.addEventListener('mousemove', function (event) {
    if (globalclicktime == 1) {
        canvas = document.getElementById("canvas");
        var offsetX = event.clientX - canvas.getBoundingClientRect().left;
        var offsetY = event.clientY - canvas.getBoundingClientRect().top;

        line = document.getElementById(arrowId);
        offsetX = parseFloat(offsetX);
        offsetY = parseFloat(offsetY);
        var pathData = line.getAttribute("d").split(" ");
        var shiftY = parseFloat(line.getAttribute("shiftY"));
        pathData[4] = (parseFloat(pathData[1]) + offsetX) / 2;
        pathData[5] = (parseFloat(pathData[2]) + offsetY) / 2 + shiftY + sign(shiftY) * Math.abs(offsetY - parseFloat(pathData[2]));
        pathData[6] = offsetX;
        pathData[7] = offsetY;
        line.setAttribute("d", pathData.join(" "));
    }
});

let input = document.querySelector('input[name="searchInfo"]');
input.addEventListener('input', function () {
    const searchText = input.value.toLowerCase();

    let draggables = document.querySelectorAll('.draggable');
    draggables.forEach(function (image) {
        const imgName = image.getAttribute('imgname').toLowerCase();
        if (imgName.includes(searchText)) {
            image.style.display = 'block';
        } else {
            image.style.display = 'none';
        }
    });
});

const card = document.getElementById('card');
cardimg = document.getElementById('cardimg');
let draggables = document.querySelectorAll(".draggable");
draggables.forEach(function (image) {

    image.addEventListener('mouseover', function (event) {
        card.style.visibility = 'visible';
        cardimg.src = "image/"+image.getAttribute('imgname')+'.png';
        let cardtitle = document.getElementById('cardtitle')
        cardtitle.textContent=image.getAttribute('imgname'); 
    });

    image.addEventListener('mouseout', function (event) {
        card.style.visibility = 'hidden';
    });
});

function selectOnclick(event) {
    tool = 0;
    document.getElementById("selecttool").style.backgroundColor = "#e6e6e6";
    document.getElementById("deletetool").style.backgroundColor = "#f2f2f2";
}

function deleteOnclick(event) {
    tool = 1;
    document.getElementById("selecttool").style.backgroundColor = "#f2f2f2";
    document.getElementById("deletetool").style.backgroundColor = "#e6e6e6";
    
}


function allowDrop(event) {
    event.preventDefault();
    console.log("dragzone");
}

function dotclick_handler(event) {
    if (globalclicktime == 0) {
        globalclicktime = 1;
        svg = document.getElementById('main-svg');
        var offsetX = (event.target.getBoundingClientRect().left - svg.getBoundingClientRect().left);
        var offsetY = (event.target.getBoundingClientRect().top - svg.getBoundingClientRect().top);
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        line.setAttribute("d", "M " + offsetX + " " + offsetY + " Q " + offsetX + " " + offsetY + " " + offsetX + " " + offsetY);
        line.setAttribute("stroke", "black"); // 设置描边颜色
        line.setAttribute("fill", "transparent");
        line.setAttribute('marker-end', 'url(#arrow)');
        line.setAttribute('marker-start', 'url(#circle)')
        line.setAttribute("stroke-width", 1);
        line.id = arrowId;
        line.setAttribute("shiftY", Math.floor(-event.target.parentNode.getBoundingClientRect().height / 2 - event.target.parentNode.getBoundingClientRect().top + event.target.getBoundingClientRect().top) + 4);
        line.style.zIndex = 2;
        line.onclick = function (event) {
            if (tool == 1) {
                deleteArrow(event.target.id);
            }
        }
        svg.appendChild(line);
        var arrowList = event.target.getAttribute('arrow-list');
        if (!arrowList) {
            var initialList = [];
            initialList.push(arrowId);
            initialList.push("start");
            event.target.setAttribute('arrow-list', JSON.stringify(initialList));

        }
        else {
            arrowList = JSON.parse(arrowList);
            arrowList.push(arrowId);
            arrowList.push("start");
            event.target.setAttribute('arrow-list', JSON.stringify(arrowList));
        }

    }
    else if (globalclicktime == 1) {
        globalclicktime = 0;
        var arrowList = event.target.getAttribute('arrow-list');
        if (!arrowList) {
            var initialList = [];
            initialList.push(arrowId);
            initialList.push("end");
            event.target.setAttribute('arrow-list', JSON.stringify(initialList));

        }
        else {
            arrowList = JSON.parse(arrowList);
            arrowList.push(arrowId);
            arrowList.push("end");
            event.target.setAttribute('arrow-list', JSON.stringify(arrowList));
        }
        var offsetX = (event.target.getBoundingClientRect().left - svg.getBoundingClientRect().left);
        var offsetY = (event.target.getBoundingClientRect().top - svg.getBoundingClientRect().top);
        line = document.getElementById(arrowId);
        var pathData = line.getAttribute("d").split(" ");
        var shiftY = parseFloat(line.getAttribute("shiftY"));
        pathData[4] = (parseFloat(pathData[1]) + offsetX) / 2;
        pathData[5] = (parseFloat(pathData[2]) + offsetY) / 2 + shiftY + sign(shiftY) * Math.abs(offsetY - parseFloat(pathData[2]));
        pathData[6] = offsetX;
        pathData[7] = offsetY;
        line.setAttribute("d", pathData.join(" "));
        line.classList.add("line");
        arrowId++;
    }
}

function updateArrow(element) {

    for (i = 1; i < 5; i++) {
        var arrowList = element.children[i].getAttribute('arrow-list');
        const svg = document.getElementById('main-svg');
        var offsetX = (element.children[i].getBoundingClientRect().left - svg.getBoundingClientRect().left);
        var offsetY = (element.children[i].getBoundingClientRect().top - svg.getBoundingClientRect().top);
        if (!arrowList) {
            continue;
        }
        else {
            arrowList = JSON.parse(arrowList);
            for (j = 0; j < arrowList.length / 2; j++) {
                if (arrowList[2 * j + 1] == "start") {
                    line = document.getElementById(arrowList[2 * j])
                    var pathData = line.getAttribute("d").split(" ");
                    var shiftY = parseFloat(line.getAttribute("shiftY"));
                    pathData[1] = offsetX;
                    pathData[2] = offsetY;
                    pathData[5] = (parseFloat(pathData[2]) + parseFloat(pathData[7])) / 2 + shiftY + sign(shiftY) * Math.abs(parseFloat(pathData[7]) - parseFloat(pathData[2]));
                    line.setAttribute("d", pathData.join(" "));
                }
                else if (arrowList[2 * j + 1] == "end") {
                    line = document.getElementById(arrowList[2 * j])
                    var pathData = line.getAttribute("d").split(" ");
                    var shiftY = parseFloat(line.getAttribute("shiftY"));
                    pathData[6] = offsetX;
                    pathData[7] = offsetY;
                    pathData[5] = (parseFloat(pathData[2]) + parseFloat(pathData[7])) / 2 + shiftY + sign(shiftY) * Math.abs(parseFloat(pathData[7]) - parseFloat(pathData[2]));
                    console.log(pathData);
                    line.setAttribute("d", pathData.join(" "));
                }
            }
        }
    }
}

function deleteArrow(id) {
    var dots = document.querySelectorAll('.dot');
    dots.forEach(dot => {
        ArrowListTemp = dot.getAttribute('arrow-list');
        if (!ArrowListTemp) {
        }
        else {
            ArrowListTemp = JSON.parse(ArrowListTemp);
            for (num = 0; num < ArrowListTemp.length / 2; num++) {
                if (ArrowListTemp[2 * num] == id) {
                    ArrowListTemp.splice(2 * num, 2);
                    console
                    dot.setAttribute('arrow-list', JSON.stringify(ArrowListTemp));
                    break;
                }
            }
        }

    });
    document.getElementById(id).remove();
}
function drop(event) {
    event.preventDefault();
    var data = event.dataTransfer.getData("text");
    var offX = event.dataTransfer.getData("offX");
    var offY = event.dataTransfer.getData("offY");
    var id = event.dataTransfer.getData('id');
    if (id == '0') {
        var element = document.createElement("div");
        let ran = Math.floor(Math.random() * 1000) + 10000;
        var image = document.createElement('img');
        image.src = data;
        image.style.position = 'relative'
        image.style.top = "2%";
        image.style.left = '2%';
        image.style.height = '96%';
        image.style.width = '96%';
        element.draggable = "true";
        element.classList.add("component");
        element.id = ran;
        element.style.width = componentWidth + 'px';
        element.style.height = componentHeight + 'px';
        element.style.position = "absolute";
        element.style.zIndex = 2;
        element.ondragstart = function (event) {
            event.dataTransfer.setData('offX', offX);
            event.dataTransfer.setData('offY', offY);
            if (event.target.parentNode.id != "canvas") {
                event.dataTransfer.setData('id', event.target.parentNode.id)
            }
            else {
                event.dataTransfer.setData('id', event.target.id)
            }
        };
        element.onclick = function (event) {
            if (tool == 1) {
                var main_div = event.target.parentNode;
                for (i = 1; i < 5; i++) {
                    var arrowList = main_div.children[i].getAttribute('arrow-list');
                    if (!arrowList) {
                        continue;
                    }
                    else {
                        arrowList = JSON.parse(arrowList);

                        for (j = 0; j < arrowList.length / 2;) {
                            console.log("j:" + j);
                            console.log("id:" + arrowList[2 * j])
                            deleteArrow(arrowList[2 * j]);
                            console.log("j:" + j)
                            console.log("deleteArrow:id:" + arrowList[2 * j])
                            j++;
                        }
                    }
                }
                main_div.remove();
            }
        }
        element.appendChild(image);
        var dot = document.createElement("div");
        dot.classList.add("dot");
        dot.classList.add("top")
        element.appendChild(dot)
        var dot = document.createElement("div");
        dot.classList.add("dot");
        dot.classList.add("bottom");
        element.appendChild(dot)
        var dot = document.createElement("div");
        dot.classList.add("dot");
        dot.classList.add("left");
        element.appendChild(dot)
        var dot = document.createElement("div");
        dot.classList.add("dot");
        dot.classList.add("right");
        element.appendChild(dot)
        element.style.left = (event.clientX - document.getElementById("canvas").getBoundingClientRect().left + Number(offX)) + 'px'; // 调整元件位置
        element.style.top = (event.clientY - document.getElementById("canvas").getBoundingClientRect().top + Number(offY)) + 'px';
        element.addEventListener('mouseover', function () {
            element.classList.add('image-hovered');
            for (i = 1; i < 5; i++) {
                element.children[i].style.visibility = 'visible';
            }
        });

        element.addEventListener('mouseout', function () {
            element.classList.remove('image-hovered');
            for (i = 1; i < 5; i++) {
                element.children[i].style.visibility = 'hidden';
            }
        });

        for (i = 1; i < 5; i++) {
            element.children[i].addEventListener('click', function (event) {
                dotclick_handler(event);
            });
        }


        event.target.parentNode.appendChild(element);

    }
    else {
        var element = document.getElementById(id);
        element.style.left = (event.clientX - document.getElementById("canvas").getBoundingClientRect().left + Number(offX)) + 'px'; // 调整元件位置
        element.style.top = (event.clientY - document.getElementById("canvas").getBoundingClientRect().top + Number(offY)) + 'px';
        updateArrow(element);
        isDarggingElement = false
    }


}


function dragstart_handler(event) {

    console.log("Startdrag");
    isDarggingElement = true;
    var offX = event.target.getBoundingClientRect().left - event.clientX;
    var offY = event.target.getBoundingClientRect().top - event.clientY;
    event.dataTransfer.setData('offX', offX);
    event.dataTransfer.setData('offY', offY);
    event.dataTransfer.setData('id', 0);
}


