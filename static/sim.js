const mapLayouts = {
    torrens: {
        ENTRANCE: { x: 90, y: 210 },
        RECEPTION: { x: 220, y: 210 },
        CORRIDOR_A: { x: 380, y: 210 },
        LIFT: { x: 540, y: 160 },
        ROOM_G12: { x: 680, y: 160 },
        EXIT: { x: 540, y: 300 }
    },
    shopping_centre: {
        ENTRANCE: { x: 100, y: 220 },
        INFO_DESK: { x: 250, y: 220 },
        CORRIDOR_B: { x: 420, y: 220 },
        FOOD_COURT: { x: 610, y: 220 },
        EXIT: { x: 420, y: 330 }
    }
};

window.drawMap = function (buildingName, path = [], nodes = {}) {
    const canvas = document.getElementById("mapCanvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const layout = mapLayouts[buildingName];
    if (!layout) return;

    resizeCanvasForScreen(canvas);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    drawBackground(ctx, canvas);
    drawConnections(ctx, buildingName, layout);
    drawPath(ctx, layout, path);
    drawNodes(ctx, layout, nodes, path);
};

function resizeCanvasForScreen(canvas) {
    const isMobile = window.innerWidth < 640;
    canvas.width = isMobile ? 900 : 760;
    canvas.height = isMobile ? 500 : 420;
}

function drawBackground(ctx, canvas) {
    ctx.fillStyle = "#f9fbfd";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "#eaf1f7";
    ctx.fillRect(50, 120, canvas.width - 100, 220);

    ctx.strokeStyle = "#c8d6e5";
    ctx.lineWidth = 2;
    ctx.strokeRect(50, 120, canvas.width - 100, 220);
}

function drawConnections(ctx, buildingName, layout) {
    const edgePairs = getEdgePairs(buildingName);

    ctx.strokeStyle = "#b7c6d8";
    ctx.lineWidth = 6;

    edgePairs.forEach(([from, to]) => {
        const start = layout[from];
        const end = layout[to];

        if (!start || !end) return;

        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
    });
}

function drawPath(ctx, layout, path) {
    if (!path || path.length < 2) return;

    ctx.strokeStyle = "#1d4ed8";
    ctx.lineWidth = 8;

    for (let i = 0; i < path.length - 1; i++) {
        const current = layout[path[i]];
        const next = layout[path[i + 1]];

        if (!current || !next) continue;

        ctx.beginPath();
        ctx.moveTo(current.x, current.y);
        ctx.lineTo(next.x, next.y);
        ctx.stroke();
    }
}

function drawNodes(ctx, layout, nodes, path) {
    Object.keys(layout).forEach(nodeId => {
        const point = layout[nodeId];
        const isActive = path.includes(nodeId);
        const isStart = path.length > 0 && path[0] === nodeId;
        const isDestination = path.length > 0 && path[path.length - 1] === nodeId;
        const label = nodes[nodeId]?.label || nodeId;

        ctx.beginPath();
        ctx.fillStyle = isDestination ? "#059669" : isActive ? "#2563eb" : "#16395f";
        ctx.arc(point.x, point.y, 18, 0, Math.PI * 2);
        ctx.fill();

        if (isStart) {
            ctx.beginPath();
            ctx.strokeStyle = "#f59e0b";
            ctx.lineWidth = 4;
            ctx.arc(point.x, point.y, 24, 0, Math.PI * 2);
            ctx.stroke();
        }

        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 11px Arial";
        ctx.textAlign = "center";
        ctx.fillText("●", point.x, point.y + 4);

        ctx.fillStyle = "#1d2733";
        ctx.font = "16px Arial";
        ctx.fillText(label, point.x, point.y - 28);
    });
}

function getEdgePairs(buildingName) {
    if (buildingName === "shopping_centre") {
        return [
            ["ENTRANCE", "INFO_DESK"],
            ["INFO_DESK", "CORRIDOR_B"],
            ["CORRIDOR_B", "FOOD_COURT"],
            ["CORRIDOR_B", "EXIT"]
        ];
    }

    return [
        ["ENTRANCE", "RECEPTION"],
        ["RECEPTION", "CORRIDOR_A"],
        ["CORRIDOR_A", "LIFT"],
        ["LIFT", "ROOM_G12"],
        ["CORRIDOR_A", "EXIT"]
    ];
}

window.addEventListener("resize", function () {
    const building = document.getElementById("building");
    if (!building) return;

    const buildingName = building.value;
    if (window.drawMap) {
        window.drawMap(buildingName, window.latestPath || [], window.latestNodes || {});
    }
});