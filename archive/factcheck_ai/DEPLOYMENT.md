# Backend Deployment Guide

This guide provides step-by-step instructions on how to deploy your extension backend to your server at `10.55.205.205`.

## Prerequisites
- Server IP: `10.55.205.205`
- SSH Password
- Your backend code located in `d:\gradproject\chrome_extention\backend`

## Step 1: Connect to the Server

Open your Windows terminal (PowerShell or Command Prompt) and connect to the server via SSH. Replace `username` with your actual server username (e.g., `ubuntu`, `root`, or your specific username).

```bash
ssh username@10.55.205.205
```

When prompted, enter your password. Note that the characters won't show up on the screen as you type.

## Step 2: Update Server and Install Dependencies

Once logged into the server, it's good practice to update the system and install Python and pip if they aren't already installed. Run these commands:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv -y
```
*(Assuming the server is running Ubuntu/Debian. If it's CentOS/RHEL, use `yum` instead).*

## Step 3: Transfer Files to the Server

Open a **new** terminal window on your local Windows machine (keep the SSH session open in the other window). Use `scp` to copy your backend folder to the server.

Navigate to your project directory:
```powershell
cd d:\gradproject\chrome_extention\
```

Copy the folder:
```powershell
scp -r backend username@10.55.205.205:~/
```
*(You will be asked for your password again).*

## Step 4: Set Up the Python Environment

Go back to your SSH terminal (where you are logged into the server) and navigate to the newly copied folder:

```bash
cd ~/backend
```

Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```
*(If you are using FastAPI, make sure `uvicorn` is in your `requirements.txt` or install it with `pip install uvicorn`)*.

## Step 5: Run the Backend

You can test if the backend runs successfully by starting it manually:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
*(Adjust `app:app` if your FastAPI instance is named differently).*

If it runs without errors, press `Ctrl+C` to stop it.

## Step 6: Keep the Backend Running in the Background (Recommended)

To keep the server running even after you close the SSH terminal, you can use `tmux`.

1. Start a new tmux session:
   ```bash
   tmux new -s mybackend
   ```
2. Navigate to your folder and activate the virtual environment:
   ```bash
   cd ~/backend
   source venv/bin/activate
   ```
3. Start your app:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
4. Detach from the session so it keeps running in the background: Press `Ctrl+b`, then release both keys, and press `d`.

*To re-attach to this session later, run `tmux attach -t mybackend`.*

## Step 7: Access the Backend

Your backend should now be accessible at:
`http://10.55.205.205:8000`

### Note on Firewalls
If you cannot access the IP from your browser, you might need to open port 8000 on your server's firewall.
For Ubuntu (UFW):
```bash
sudo ufw allow 8000
```
