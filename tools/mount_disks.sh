#!/bin/bash
# Mount all data disks and add to fstab
set -e

echo "=== Mounting data disks ==="

mount /dev/sda1 /home/tradebot/ssd_data_1
echo "  sda1 (Crucial 932GB SSD) → ssd_data_1"

mount /dev/sdd2 /home/tradebot/ssd_data_2
echo "  sdd2 (Samsung 466GB SSD) → ssd_data_2"

mount /dev/sdc1 /home/tradebot/data_1
echo "  sdc1 (Samsung 932GB HDD) → data_1"

mount /dev/sde1 /home/tradebot/data_2
echo "  sde1 (WD Red 1.8TB HDD) → data_2"

echo ""
echo "=== Setting ownership ==="
chown tradebot:tradebot /home/tradebot/ssd_data_1
chown tradebot:tradebot /home/tradebot/ssd_data_2
chown tradebot:tradebot /home/tradebot/data_1
chown tradebot:tradebot /home/tradebot/data_2
echo "  Done"

echo ""
echo "=== Adding to fstab ==="
grep -q "ssd_data_1" /etc/fstab || echo "UUID=a08ef243-8502-4b05-9b0c-26a076cd7497 /home/tradebot/ssd_data_1 ext4 defaults,noatime 0 2" >> /etc/fstab
grep -q "ssd_data_2" /etc/fstab || echo "UUID=7ef78b6b-81fd-4c58-84d4-3dccd88a8f79 /home/tradebot/ssd_data_2 ext4 defaults,noatime 0 2" >> /etc/fstab
grep -q "data_1" /etc/fstab || echo "UUID=3f593d4d-02d0-4fb8-86f5-c9bb571f3855 /home/tradebot/data_1 ext4 defaults,noatime 0 2" >> /etc/fstab
grep -q "data_2" /etc/fstab || echo "UUID=f96a9cc7-cf41-4997-a6cf-23499bfab6fe /home/tradebot/data_2 ext4 defaults,noatime 0 2" >> /etc/fstab
echo "  fstab updated"

echo ""
echo "=== Disk space ==="
df -h /home/tradebot/ssd_data_1 /home/tradebot/ssd_data_2 /home/tradebot/data_1 /home/tradebot/data_2

echo ""
echo "Done! Disks will auto-mount after reboot."
