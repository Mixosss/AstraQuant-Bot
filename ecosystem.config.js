module.exports = {
  apps : [{
    name      : 'AlphaBot',
    script    : 'main.py',
    interpreter: 'python3',
    interpreter_args: '-u',
    treekill  : true,
    kill_timeout : 3000
  }]
};