import subprocess
import sys
import os

def main():
    print("=" * 70)
    print(" LAUNCHING UAV-NEUROSYM GROUND CONTROL STATION DASHBOARD")
    print("=" * 70)
    
    script_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", script_path]
    
    print(f"[+] Executing: {' '.join(cmd)}")
    print("[+] Opening web browser at http://localhost:8501 ...\n")
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
