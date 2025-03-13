// import * as d3 from 'd3';

// function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
//     container, node?: nn.Node) {
//   let x = cx - RECT_SIZE / 2;
//   let y = cy - RECT_SIZE / 2;

//   let nodeGroup = container.append("g")
//     .attr({
//       "class": "node",
//       "id": `node${nodeId}`,
//       "transform": `translate(${x},${y})`
//     });

//   // Draw the main rectangle.
//   nodeGroup.append("rect")
//     .attr({
//       x: 0,
//       y: 0,
//       width: RECT_SIZE,
//       height: RECT_SIZE,
//     });
//   let activeOrNotClass = state[nodeId] ? "active" : "inactive";
//   if (isInput) {
//     let label = INPUTS[nodeId].label != null ?
//         INPUTS[nodeId].label : nodeId;
//     // Draw the input label.
//     let text = nodeGroup.append("text").attr({
//       class: "main-label",
//       x: -10,
//       y: RECT_SIZE / 2, "text-anchor": "end"
//     });
//     if (/[_^]/.test(label)) {
//       let myRe = /(.*?)([_^])(.)/g;
//       let myArray;
//       let lastIndex;
//       while ((myArray = myRe.exec(label)) != null) {
//         lastIndex = myRe.lastIndex;
//         let prefix = myArray[1];
//         let sep = myArray[2];
//         let suffix = myArray[3];
//         if (prefix) {
//           text.append("tspan").text(prefix);
//         }
//         text.append("tspan")
//         .attr("baseline-shift", sep === "_" ? "sub" : "super")
//         .style("font-size", "9px")
//         .text(suffix);
//       }
//       if (label.substring(lastIndex)) {
//         text.append("tspan").text(label.substring(lastIndex));
//       }
//     } else {
//       text.append("tspan").text(label);
//     }
//     nodeGroup.classed(activeOrNotClass, true);
//   }
//   if (!isInput) {
//     // Draw the node's bias.
//     nodeGroup.append("rect")
//       .attr({
//         id: `bias-${nodeId}`,
//         x: -BIAS_SIZE - 2,
//         y: RECT_SIZE - BIAS_SIZE + 3,
//         width: BIAS_SIZE,
//         height: BIAS_SIZE,
//       }).on("mouseenter", function() {
//         updateHoverCard(HoverType.BIAS, node, d3.mouse(container.node()));
//       }).on("mouseleave", function() {
//         updateHoverCard(null);
//       });
//   }

//   // Draw the node's canvas.
//   let div = d3.select("#network").insert("div", ":first-child")
//     .attr({
//       "id": `canvas-${nodeId}`,
//       "class": "canvas"
//     })
//     .style({
//       position: "absolute",
//       left: `${x + 3}px`,
//       top: `${y + 3}px`
//     })
//     .on("mouseenter", function() {
//       selectedNodeId = nodeId;
//       div.classed("hovered", true);
//       nodeGroup.classed("hovered", true);
//       updateDecisionBoundary(network, false);
//       heatMap.updateBackground(boundary[nodeId], state.discretize);
//     })
//     .on("mouseleave", function() {
//       selectedNodeId = null;
//       div.classed("hovered", false);
//       nodeGroup.classed("hovered", false);
//       updateDecisionBoundary(network, false);
//       heatMap.updateBackground(boundary[nn.getOutputNode(network).id],
//           state.discretize);
//     });
//   if (isInput) {
//     div.on("click", function() {
//       state[nodeId] = !state[nodeId];
//       parametersChanged = true;
//       reset();
//     });
//     div.style("cursor", "pointer");
//   }
//   if (isInput) {
//     div.classed(activeOrNotClass, true);
//   }
//   let nodeHeatMap = new HeatMap(RECT_SIZE, DENSITY / 10, xDomain,
//       xDomain, div, {noSvg: true});
//   div.datum({heatmap: nodeHeatMap, id: nodeId});

// }

console.log("Hello from node js");