# 智慧交通 JSON 协议 v2.0

本文件是字段说明；可执行校验规则见 `traffic_message.schema.json`。

## 传输封装

- 字符编码：UTF-8。
- 一条消息对应一行，消息以 `\n` 结束。
- 悟净 UART 输出：`ICRAFT_JSON:<JSON>\n`。
- ALINX 到 PC TCP 输出：`<JSON>\n`，不包含 `ICRAFT_JSON:` 前缀。
- JSON 必须是单行紧凑格式，不允许在一个对象内部换行。

## 固定结构

```json
{"schema_version":"2.0","timestamp":"2026-08-12 14:30:00","scene_id":"camera_0","frame_id":15200,"objects":[{"track_id":23,"class":"car","bbox":[320.5,210.0,180.0,96.0],"score":0.91}],"events":[{"event_type":"collision","participants":[23,31],"participant_classes":["car","car"],"location":[810.0,460.0],"desc":"汽车与汽车发生碰撞。"}],"traffic_stats":{"window_seconds":10,"count_north":12,"count_south":8,"count_east":15,"count_west":6,"window_north":2,"window_south":1,"window_east":3,"window_west":1,"north_south_total":20,"east_west_total":21,"vehicles_in_scene":9},"traffic_light":{"control_mode":"simulation","hardware_output":false,"current_phase":"north_south_green","north":"green","south":"green","east":"red","west":"red","remaining_seconds":35,"planned_green_seconds":{"north_south":55,"east_west":60},"adjustment":{"changed":true,"reason":"east_west_flow_higher","previous_north_south_green":40,"new_north_south_green":55,"previous_east_west_green":40,"new_east_west_green":60}}}
```

## 字段约束

| 字段 | 类型 | 约束 |
|---|---|---|
| `schema_version` | string | 固定为 `2.0` |
| `timestamp` | string | 本地时间，格式 `YYYY-MM-DD HH:MM:SS` |
| `scene_id` | string | 当前为 `camera_<source_id>` |
| `frame_id` | integer | 板端递增帧号 |
| `objects` | array | 当前统计窗口代表帧中的跟踪目标 |
| `events` | array | 无事故时为空；有事故时保留原碰撞结构 |
| `traffic_stats.window_seconds` | integer | 本次报告窗口长度 |
| `count_*` | integer | 程序启动后的累计跨线计数 |
| `window_*` | integer | 当前报告窗口内的跨线计数 |
| `vehicles_in_scene` | integer | 当前帧内已跟踪车辆数 |
| `control_mode` | string | 固定为 `simulation` |
| `hardware_output` | boolean | 固定为 `false` |
| `current_phase` | string | 六状态之一 |
| `north/south/east/west` | string | `red`、`yellow` 或 `green` |
| `remaining_seconds` | integer | 当前相位剩余秒数 |
| `planned_green_seconds` | object | 下一次进入相应方向绿灯时采用的计划时长，不中途改变当前绿灯 |
| `adjustment.changed` | boolean | 本窗口是否调整绿灯计划 |
| `adjustment.reason` | string | 流量比较产生的固定枚举值 |

`current_phase` 的合法值：`north_south_green`、`north_south_yellow`、
`all_red_to_east_west`、`east_west_green`、`east_west_yellow`、
`all_red_to_north_south`。

新增字段时只能向后追加，不能删除或修改上述字段的类型。PC 端必须通过
`schema_version` 选择解析逻辑，并容忍未知字段。
